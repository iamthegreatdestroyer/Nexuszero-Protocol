/**
 * Cryptographic operations module
 * 
 * This module provides low-level cryptographic operations for the SDK.
 * In production, these would call into native Rust FFI bindings.
 */

import {
  Commitment,
  CryptoParameters,
  ErrorCode,
  NexuszeroError,
  SecurityLevel,
} from "./types";

/**
 * Generate a random blinding factor for commitments
 * @param length - Length in bytes (default: 32)
 * @returns Random bytes
 */
export function generateBlinding(length: number = 32): Uint8Array {
  if (typeof crypto !== "undefined" && crypto.getRandomValues) {
    // Browser environment
    return crypto.getRandomValues(new Uint8Array(length));
  } else if (typeof require !== "undefined") {
    // Node.js environment
    try {
      const nodeCrypto = require("crypto");
      return new Uint8Array(nodeCrypto.randomBytes(length));
    } catch (error) {
      throw new NexuszeroError(
        ErrorCode.FFIError,
        "Failed to generate random bytes",
        error
      );
    }
  } else {
    throw new NexuszeroError(
      ErrorCode.FFIError,
      "No secure random source available"
    );
  }
}

/**
 * Get cryptographic parameters for a security level
 * @param securityLevel - Desired security level
 * @returns Cryptographic parameters
 */
export function getSecurityParameters(
  securityLevel: SecurityLevel
): CryptoParameters {
  switch (securityLevel) {
    case SecurityLevel.Bit128:
      return {
        n: 1024,
        q: 12289,
        sigma: 3.2,
        securityLevel: SecurityLevel.Bit128,
      };
    case SecurityLevel.Bit192:
      return {
        n: 2048,
        q: 40961,
        sigma: 3.2,
        securityLevel: SecurityLevel.Bit192,
      };
    case SecurityLevel.Bit256:
      return {
        n: 4096,
        q: 65537,
        sigma: 3.2,
        securityLevel: SecurityLevel.Bit256,
      };
    default:
      throw new NexuszeroError(
        ErrorCode.InvalidParameters,
        `Invalid security level: ${securityLevel}`
      );
  }
}

/**
 * Validate cryptographic parameters
 * @param params - Parameters to validate
 * @throws {NexuszeroError} If parameters are invalid
 */
export function validateParameters(params: CryptoParameters): void {
  if (params.n <= 0 || !Number.isInteger(params.n)) {
    throw new NexuszeroError(
      ErrorCode.InvalidParameters,
      "Dimension (n) must be a positive integer"
    );
  }

  if (params.q <= 0 || !Number.isInteger(params.q)) {
    throw new NexuszeroError(
      ErrorCode.InvalidParameters,
      "Modulus (q) must be a positive integer"
    );
  }

  if (params.sigma <= 0) {
    throw new NexuszeroError(
      ErrorCode.InvalidParameters,
      "Sigma must be positive"
    );
  }

  // Check if n is a power of 2 (required for NTT)
  if ((params.n & (params.n - 1)) !== 0) {
    throw new NexuszeroError(
      ErrorCode.InvalidParameters,
      "Dimension (n) must be a power of 2"
    );
  }
}

/**
 * Create a Pedersen commitment to a value
 * 
 * NOTE: This is a stub implementation. In production, this would call
 * the native Rust library via FFI bindings.
 * 
 * @param value - Value to commit to
 * @param blinding - Blinding factor (generated if not provided)
 * @returns Commitment
 */
export async function createCommitment(
  value: bigint,
  blinding?: Uint8Array
): Promise<Commitment> {
  // Generate blinding if not provided
  const blindingFactor = blinding || generateBlinding(32);

  // TODO: Replace with actual FFI call to Rust library
  // For now, create a mock commitment
  const commitmentData = new Uint8Array(64);
  
  // Simple mock: hash value and blinding together
  const valueBytes = bigIntToBytes(value);
  for (let i = 0; i < Math.min(valueBytes.length, 32); i++) {
    commitmentData[i] = valueBytes[i];
  }
  for (let i = 0; i < Math.min(blindingFactor.length, 32); i++) {
    commitmentData[i + 32] = blindingFactor[i];
  }

  return {
    data: commitmentData,
    value,
    blinding: blindingFactor,
  };
}

/**
 * Verify that a commitment opens to a specific value
 * 
 * @param commitment - Commitment to verify
 * @param value - Value that should be committed
 * @param blinding - Blinding factor used
 * @returns True if commitment is valid
 */
export async function verifyCommitment(
  commitment: Commitment,
  value: bigint,
  blinding: Uint8Array
): Promise<boolean> {
  // TODO: Replace with actual FFI call to Rust library
  // For now, recreate the commitment and compare
  const recomputed = await createCommitment(value, blinding);
  
  if (recomputed.data.length !== commitment.data.length) {
    return false;
  }

  for (let i = 0; i < commitment.data.length; i++) {
    if (commitment.data[i] !== recomputed.data[i]) {
      return false;
    }
  }

  return true;
}

/**
 * Convert a bigint to bytes (little-endian)
 * @param value - BigInt value
 * @returns Byte array
 */
function bigIntToBytes(value: bigint): Uint8Array {
  const hex = value.toString(16).padStart(64, "0");
  const bytes = new Uint8Array(32);
  for (let i = 0; i < 32; i++) {
    bytes[i] = parseInt(hex.slice(i * 2, i * 2 + 2), 16);
  }
  return bytes;
}

/**
 * Convert bytes to a bigint (little-endian)
 * @param bytes - Byte array
 * @returns BigInt value
 */
export function bytesToBigInt(bytes: Uint8Array): bigint {
  let result = 0n;
  for (let i = bytes.length - 1; i >= 0; i--) {
    result = (result << 8n) + BigInt(bytes[i]);
  }
  return result;
}
