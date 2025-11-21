# Week 1: Cryptography Module - Copilot Prompts

**Project:** Nexuszero Protocol  
**Phase:** Week 1 - Lattice-Based Cryptography Foundation  
**Duration:** 7 days  
**Goal:** Establish secure, quantum-resistant cryptographic primitives

---

## 📋 DAILY BREAKDOWN

### Day 1-2: Lattice-Based Crypto Library Initialization
### Day 3-4: Proof Structures (Statement, Witness, Proof)
### Day 5: Parameter Selection Algorithms
### Day 6-7: Unit Tests with Test Vectors

---

## 🔐 DAY 1-2: LATTICE-BASED CRYPTO LIBRARY

### Prompt 1.1: Project Structure & Dependencies

```
Create a Rust project structure for a lattice-based cryptography library with the following requirements:

## Project Requirements
- **Name:** nexuszero-crypto
- **Type:** Rust library crate
- **Target:** Quantum-resistant zero-knowledge proof system
- **Focus:** Learning With Errors (LWE) and Ring-LWE primitives

## Structure to Create
```
nexuszero-crypto/
├── Cargo.toml
├── src/
│   ├── lib.rs                  # Main library entry
│   ├── lattice/
│   │   ├── mod.rs              # Lattice operations module
│   │   ├── lwe.rs              # LWE primitives
│   │   ├── ring_lwe.rs         # Ring-LWE operations
│   │   └── sampling.rs         # Error distribution sampling
│   ├── proof/
│   │   ├── mod.rs              # Proof system module
│   │   ├── statement.rs        # Statement structures
│   │   ├── witness.rs          # Witness structures
│   │   └── proof.rs            # Proof generation/verification
│   ├── params/
│   │   ├── mod.rs              # Parameter module
│   │   └── security.rs         # Security level configurations
│   └── utils/
│       ├── mod.rs              # Utility functions
│       └── math.rs             # Mathematical primitives
├── tests/
│   ├── integration_tests.rs
│   └── test_vectors/
│       └── nist_vectors.json
└── benches/
    └── crypto_benchmarks.rs
```

## Dependencies to Add (Cargo.toml)
- **ndarray** = "0.15" (matrix operations)
- **rand** = "0.8" (random number generation)
- **sha3** = "0.10" (hashing for Fiat-Shamir)
- **num-bigint** = "0.4" (large integer arithmetic)
- **serde** = { version = "1.0", features = ["derive"] } (serialization)
- **criterion** = "0.5" (benchmarking)

## Initial Code Requirements
1. Create basic module structure with proper visibility
2. Add placeholder trait definitions for key operations:
   - `LatticeParameters` trait
   - `ProofSystem` trait
   - `SecurityLevel` enum (128-bit, 192-bit, 256-bit)
3. Set up error handling with custom `CryptoError` type
4. Configure workspace with development profiles

## Output Deliverables
- Complete Cargo.toml with all dependencies
- Module structure with mod.rs files
- Trait definitions in lib.rs
- Basic error types
- README.md with project overview

Generate the complete project initialization following Rust best practices.
```

---

### Prompt 1.2: LWE Primitive Implementation

```
Implement Learning With Errors (LWE) cryptographic primitives for the nexuszero-crypto library.

## Background: LWE Problem
The LWE problem forms the security foundation for our quantum-resistant ZK proofs:
- Given (A, b = As + e mod q), recover secret vector s
- A is a random matrix, e is small error vector
- Security reduces to worst-case lattice problems (SIVP)

## Implementation Requirements

### 1. Core LWE Structure (src/lattice/lwe.rs)

Create the following types and implementations:

**LWEParameters struct:**
- n: usize (dimension, security parameter)
- m: usize (number of samples)
- q: u64 (modulus)
- sigma: f64 (error distribution standard deviation)

**LWEPublicKey struct:**
- A: Matrix (m × n matrix over Z_q)
- b: Vector (m-dimensional vector over Z_q)

**LWESecretKey struct:**
- s: Vector (n-dimensional secret vector)

**LWECiphertext struct:**
- u: Vector (ciphertext component 1)
- v: u64 (ciphertext component 2)

### 2. Key Operations to Implement

```rust
// Key generation
fn keygen(params: &LWEParameters) -> (LWESecretKey, LWEPublicKey);

// Encryption: Enc(pk, m) where m ∈ {0,1}
fn encrypt(pk: &LWEPublicKey, message: bool, params: &LWEParameters) -> LWECiphertext;

// Decryption: Dec(sk, ct) -> m
fn decrypt(sk: &LWESecretKey, ct: &LWECiphertext, params: &LWEParameters) -> bool;

// Sample error from discrete Gaussian
fn sample_error(sigma: f64, dimension: usize) -> Vec<i64>;

// Matrix-vector operations mod q
fn matrix_vector_mult_mod(A: &Matrix, v: &Vec<i64>, q: u64) -> Vec<i64>;
```

### 3. Error Sampling (src/lattice/sampling.rs)

Implement discrete Gaussian sampling:
- Use Box-Muller transform for continuous Gaussian
- Round to nearest integer for discrete Gaussian
- Rejection sampling for tail-cut Gaussian
- Verify samples have correct standard deviation

### 4. Security Considerations

Include:
- Constant-time operations where possible
- Secure randomness from `rand::CryptoRng`
- Parameter validation (check q is prime, n is power of 2)
- Bounds checking on all arithmetic

### 5. Unit Tests

Create tests for:
- Correctness: Encrypt then decrypt recovers message
- Homomorphism: Verify additive homomorphic property
- Error distribution: Statistical tests on sampled errors
- Edge cases: Zero message, maximum parameters

## Example Test Structure
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lwe_correctness() {
        let params = LWEParameters::new(256, 512, 12289, 3.2);
        let (sk, pk) = keygen(&params);
        
        let m0 = false;
        let ct0 = encrypt(&pk, m0, &params);
        assert_eq!(decrypt(&sk, &ct0, &params), m0);
        
        let m1 = true;
        let ct1 = encrypt(&pk, m1, &params);
        assert_eq!(decrypt(&sk, &ct1, &params), m1);
    }
    
    #[test]
    fn test_error_distribution() {
        // Verify sampled errors follow Gaussian distribution
    }
}
```

## Performance Requirements
- Key generation: < 50ms for 128-bit security
- Encryption: < 5ms per bit
- Decryption: < 5ms per bit
- Use ndarray for optimized matrix operations

Implement complete LWE system with all required functions, tests, and documentation.
```

---

### Prompt 1.3: Ring-LWE Implementation

```
Implement Ring Learning With Errors (Ring-LWE) for more efficient operations in nexuszero-crypto.

## Background: Ring-LWE vs Standard LWE
Ring-LWE operates in polynomial rings R_q = Z_q[X]/(X^n + 1) where:
- n is a power of 2 (typically 256, 512, 1024)
- Operations are polynomial multiplication mod (X^n + 1)
- ~10x more efficient than standard LWE
- Security based on Ring-SIS and Ring-LWE problems

## Implementation Requirements

### 1. Polynomial Ring Structure (src/lattice/ring_lwe.rs)

**Polynomial struct:**
```rust
pub struct Polynomial {
    coeffs: Vec<i64>,  // Coefficients [a_0, a_1, ..., a_{n-1}]
    degree: usize,      // Degree n (power of 2)
    modulus: u64,       // Coefficient modulus q
}
```

**RingLWEParameters:**
- n: usize (polynomial degree, must be power of 2)
- q: u64 (coefficient modulus, typically prime)
- sigma: f64 (error distribution parameter)
- phi: Polynomial (cyclotomic polynomial X^n + 1)

### 2. Core Ring Operations

Implement the following polynomial operations:

```rust
// Polynomial addition in R_q
fn poly_add(a: &Polynomial, b: &Polynomial, q: u64) -> Polynomial;

// Polynomial subtraction in R_q
fn poly_sub(a: &Polynomial, b: &Polynomial, q: u64) -> Polynomial;

// Polynomial multiplication mod (X^n + 1) using NTT
fn poly_mult_ntt(a: &Polynomial, b: &Polynomial, q: u64) -> Polynomial;

// Number Theoretic Transform (NTT) for fast multiplication
fn ntt(poly: &Polynomial, q: u64, primitive_root: u64) -> Vec<i64>;
fn intt(transformed: &Vec<i64>, q: u64, primitive_root: u64) -> Polynomial;

// Scalar multiplication
fn poly_scalar_mult(poly: &Polynomial, scalar: i64, q: u64) -> Polynomial;

// Sample polynomial with coefficients from error distribution
fn sample_poly_error(n: usize, sigma: f64, q: u64) -> Polynomial;
```

### 3. Ring-LWE Cryptosystem

**Types:**
```rust
pub struct RingLWESecretKey {
    s: Polynomial,  // Secret polynomial
}

pub struct RingLWEPublicKey {
    a: Polynomial,  // Random polynomial
    b: Polynomial,  // b = a*s + e (mod q, mod X^n+1)
}

pub struct RingLWECiphertext {
    u: Polynomial,  // u = a*r + e1
    v: Polynomial,  // v = b*r + e2 + encode(m)
}
```

**Key Operations:**
```rust
fn ring_keygen(params: &RingLWEParameters) -> (RingLWESecretKey, RingLWEPublicKey);

fn ring_encrypt(
    pk: &RingLWEPublicKey, 
    message: &[bool],  // Message bits to encrypt
    params: &RingLWEParameters
) -> RingLWECiphertext;

fn ring_decrypt(
    sk: &RingLWESecretKey, 
    ct: &RingLWECiphertext, 
    params: &RingLWEParameters
) -> Vec<bool>;

// Encode message bits into polynomial
fn encode_message(message: &[bool], n: usize, q: u64) -> Polynomial;

// Decode polynomial back to message bits
fn decode_message(poly: &Polynomial) -> Vec<bool>;
```

### 4. NTT Optimization

Implement Number Theoretic Transform for O(n log n) polynomial multiplication:

```rust
// Find primitive nth root of unity mod q
fn find_primitive_root(n: usize, q: u64) -> Option<u64>;

// Precompute twiddle factors for NTT
fn precompute_twiddles(n: usize, root: u64, q: u64) -> Vec<u64>;

// Fast polynomial multiplication using NTT
fn ntt_mult(a: &Polynomial, b: &Polynomial, params: &RingLWEParameters) -> Polynomial;
```

### 5. Parameter Selection

Create standard parameter sets:

```rust
impl RingLWEParameters {
    // 128-bit security (NIST Level 1)
    pub fn new_128bit_security() -> Self {
        RingLWEParameters {
            n: 512,
            q: 12289,  // Prime, q ≡ 1 (mod 2n)
            sigma: 3.2,
            phi: Polynomial::cyclotomic(512),
        }
    }
    
    // 192-bit security (NIST Level 3)
    pub fn new_192bit_security() -> Self { /* ... */ }
    
    // 256-bit security (NIST Level 5)
    pub fn new_256bit_security() -> Self { /* ... */ }
}
```

### 6. Comprehensive Tests

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_ring_lwe_correctness() {
        // Encrypt/decrypt multiple message bits
    }
    
    #[test]
    fn test_polynomial_arithmetic() {
        // Verify ring operations
    }
    
    #[test]
    fn test_ntt_correctness() {
        // NTT(INTT(x)) = x
    }
    
    #[test]
    fn test_ntt_multiplication() {
        // Compare NTT mult vs naive mult
    }
    
    #[test]
    fn test_cyclotomic_reduction() {
        // Verify X^n + 1 reduction
    }
}
```

### 7. Benchmarks

Create benchmarks comparing:
- NTT multiplication vs schoolbook multiplication
- Ring-LWE vs standard LWE encryption speed
- Key generation time for different security levels

## Performance Targets
- NTT transform: < 1ms for n=512
- Ring-LWE encryption: < 2ms for 256 message bits
- 5-10x faster than standard LWE

Implement complete Ring-LWE system with optimized NTT multiplication.
```

---

## 📝 DAY 3-4: PROOF STRUCTURES

### Prompt 2.1: Statement Structure

```
Design and implement the Statement structure for zero-knowledge proofs in nexuszero-crypto.

## Conceptual Background
A Statement represents the PUBLIC claim being proven:
- "I know x such that f(x) = y" → Statement is (f, y)
- Verifier can see Statement but NOT the witness
- Must be efficiently serializable for on-chain verification

## Implementation Requirements

### 1. Core Statement Types (src/proof/statement.rs)

```rust
use serde::{Deserialize, Serialize};

/// Represents different types of statements that can be proven
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum StatementType {
    /// Prove knowledge of discrete log: g^x = h
    DiscreteLog {
        generator: Vec<u8>,
        public_value: Vec<u8>,
    },
    
    /// Prove knowledge of hash preimage: H(x) = y
    Preimage {
        hash_function: HashFunction,
        hash_output: Vec<u8>,
    },
    
    /// Prove x ∈ [min, max] (range proof)
    Range {
        min: u64,
        max: u64,
        commitment: Vec<u8>,
    },
    
    /// Prove knowledge of signature on message
    Signature {
        public_key: Vec<u8>,
        message: Vec<u8>,
    },
    
    /// Custom statement with arbitrary circuit
    Circuit {
        circuit_hash: [u8; 32],  // Hash of circuit description
        public_inputs: Vec<Vec<u8>>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum HashFunction {
    SHA256,
    SHA3_256,
    Blake3,
}

/// Complete statement for a zero-knowledge proof
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Statement {
    /// Type and parameters of the statement
    pub statement_type: StatementType,
    
    /// Cryptographic parameters (LWE/Ring-LWE)
    pub crypto_params: CryptoParameters,
    
    /// Context information (optional)
    pub context: Option<ProofContext>,
    
    /// Version for future compatibility
    pub version: u8,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CryptoParameters {
    /// Security level (128, 192, 256 bits)
    pub security_level: SecurityLevel,
    
    /// Ring-LWE parameters
    pub ring_params: RingLWEParameters,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProofContext {
    /// Unique identifier for this proof session
    pub session_id: [u8; 16],
    
    /// Timestamp
    pub timestamp: u64,
    
    /// Application-specific metadata
    pub metadata: Vec<u8>,
}
```

### 2. Statement Builder Pattern

```rust
/// Builder for constructing statements
pub struct StatementBuilder {
    statement_type: Option<StatementType>,
    security_level: SecurityLevel,
    context: Option<ProofContext>,
}

impl StatementBuilder {
    pub fn new() -> Self { /* ... */ }
    
    pub fn discrete_log(
        mut self, 
        generator: Vec<u8>, 
        public_value: Vec<u8>
    ) -> Self { /* ... */ }
    
    pub fn preimage(
        mut self, 
        hash_fn: HashFunction, 
        hash_output: Vec<u8>
    ) -> Self { /* ... */ }
    
    pub fn range(
        mut self, 
        min: u64, 
        max: u64, 
        commitment: Vec<u8>
    ) -> Self { /* ... */ }
    
    pub fn with_security_level(mut self, level: SecurityLevel) -> Self { /* ... */ }
    
    pub fn with_context(mut self, context: ProofContext) -> Self { /* ... */ }
    
    pub fn build(self) -> Result<Statement, StatementError> { /* ... */ }
}
```

### 3. Statement Operations

```rust
impl Statement {
    /// Validate statement consistency
    pub fn validate(&self) -> Result<(), StatementError> {
        // Check security parameters are valid
        // Verify statement-specific constraints
        // Ensure all required fields are present
    }
    
    /// Serialize to bytes for transmission
    pub fn to_bytes(&self) -> Vec<u8> {
        // Use bincode or similar for efficient serialization
    }
    
    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, StatementError> {
        // Deserialize and validate
    }
    
    /// Compute cryptographic hash of statement
    pub fn hash(&self) -> [u8; 32] {
        // Hash for Fiat-Shamir transform
    }
    
    /// Estimate proof size for this statement
    pub fn estimate_proof_size(&self) -> usize {
        // Return estimated bytes for proof
    }
}
```

### 4. Error Handling

```rust
#[derive(Debug, thiserror::Error)]
pub enum StatementError {
    #[error("Invalid security level")]
    InvalidSecurityLevel,
    
    #[error("Missing required field: {0}")]
    MissingField(String),
    
    #[error("Invalid statement type")]
    InvalidStatementType,
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    #[error("Validation failed: {0}")]
    ValidationFailed(String),
}
```

### 5. Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_statement_builder() {
        let stmt = StatementBuilder::new()
            .preimage(HashFunction::SHA256, vec![0u8; 32])
            .with_security_level(SecurityLevel::Bit128)
            .build()
            .unwrap();
        
        assert!(stmt.validate().is_ok());
    }

    #[test]
    fn test_statement_serialization() {
        let stmt = /* create statement */;
        let bytes = stmt.to_bytes();
        let recovered = Statement::from_bytes(&bytes).unwrap();
        assert_eq!(stmt.hash(), recovered.hash());
    }

    #[test]
    fn test_statement_validation() {
        // Test various invalid statements
    }
}
```

### 6. Documentation Requirements

Add comprehensive docs explaining:
- How to construct each statement type
- Security considerations
- Size estimates for on-chain storage
- Examples for common use cases

## Design Goals
- ✅ Type-safe statement construction
- ✅ Efficient serialization (< 1KB for typical statements)
- ✅ Extensible for future statement types
- ✅ Clear error messages
- ✅ Zero-copy deserialization where possible

Implement complete Statement system with builder pattern and validation.
```

---

### Prompt 2.2: Witness Structure

```
Implement the Witness structure for zero-knowledge proofs - the SECRET information known only to the prover.

## Critical Security Requirements
⚠️ **WITNESS MUST NEVER BE TRANSMITTED OR STORED INSECURELY**
- Kept only in prover's memory
- Zeroed after proof generation
- No serialization to disk without encryption
- Constant-time operations to prevent timing attacks

## Implementation Requirements

### 1. Core Witness Structure (src/proof/witness.rs)

```rust
use zeroize::{Zeroize, ZeroizeOnDrop};

/// Represents SECRET knowledge that proves a statement
/// 
/// SECURITY: This structure is zeroized on drop to prevent
/// sensitive data from remaining in memory.
#[derive(Zeroize, ZeroizeOnDrop)]
pub struct Witness {
    /// The secret value(s) being proven
    secret_data: SecretData,
    
    /// Randomness used in proof (also secret)
    randomness: Vec<u8>,
    
    /// Type indicator (matches Statement type)
    witness_type: WitnessType,
}

/// Secret data for different witness types
#[derive(Zeroize, ZeroizeOnDrop)]
enum SecretData {
    /// Discrete log witness: the exponent x where g^x = h
    DiscreteLog(Vec<u8>),
    
    /// Preimage witness: x where H(x) = y
    Preimage(Vec<u8>),
    
    /// Range proof witness: (value, blinding_factor)
    Range {
        value: u64,
        blinding: Vec<u8>,
    },
    
    /// Signature witness: the signature itself
    Signature(Vec<u8>),
    
    /// Circuit witness: private inputs to circuit
    Circuit {
        private_inputs: Vec<Vec<u8>>,
    },
}

#[derive(Clone, Copy, Debug)]
enum WitnessType {
    DiscreteLog,
    Preimage,
    Range,
    Signature,
    Circuit,
}
```

### 2. Witness Construction

```rust
impl Witness {
    /// Create witness for discrete log proof
    pub fn discrete_log(exponent: Vec<u8>) -> Self {
        let mut randomness = vec![0u8; 32];
        rand::thread_rng().fill_bytes(&mut randomness);
        
        Witness {
            secret_data: SecretData::DiscreteLog(exponent),
            randomness,
            witness_type: WitnessType::DiscreteLog,
        }
    }
    
    /// Create witness for hash preimage proof
    pub fn preimage(preimage: Vec<u8>) -> Self {
        let mut randomness = vec![0u8; 32];
        rand::thread_rng().fill_bytes(&mut randomness);
        
        Witness {
            secret_data: SecretData::Preimage(preimage),
            randomness,
            witness_type: WitnessType::Preimage,
        }
    }
    
    /// Create witness for range proof
    pub fn range(value: u64, blinding: Vec<u8>) -> Self {
        let mut randomness = vec![0u8; 32];
        rand::thread_rng().fill_bytes(&mut randomness);
        
        Witness {
            secret_data: SecretData::Range { value, blinding },
            randomness,
            witness_type: WitnessType::Range,
        }
    }
    
    // Additional constructors for other witness types...
}
```

### 3. Witness Validation

```rust
impl Witness {
    /// Validate witness satisfies the statement
    /// 
    /// This checks that the witness actually proves the statement
    /// without revealing the witness itself.
    pub fn satisfies_statement(&self, statement: &Statement) -> bool {
        match (&self.secret_data, &statement.statement_type) {
            (SecretData::DiscreteLog(exp), StatementType::DiscreteLog { generator, public_value }) => {
                // Verify g^exp = public_value
                self.verify_discrete_log(generator, exp, public_value)
            },
            
            (SecretData::Preimage(pre), StatementType::Preimage { hash_function, hash_output }) => {
                // Verify H(pre) = hash_output
                self.verify_preimage(hash_function, pre, hash_output)
            },
            
            (SecretData::Range { value, blinding }, StatementType::Range { min, max, commitment }) => {
                // Verify value in range AND commitment is correct
                *value >= *min && *value <= *max && 
                    self.verify_commitment(value, blinding, commitment)
            },
            
            _ => false,  // Mismatched witness and statement types
        }
    }
    
    /// Get witness type
    pub fn witness_type(&self) -> WitnessType {
        self.witness_type
    }
    
    /// Access randomness (used in proof generation)
    pub(crate) fn randomness(&self) -> &[u8] {
        &self.randomness
    }
}
```

### 4. Secure Helper Functions

```rust
impl Witness {
    /// Verify discrete log in constant time
    fn verify_discrete_log(&self, g: &[u8], x: &[u8], h: &[u8]) -> bool {
        // Implement modular exponentiation g^x mod p
        // Compare with h in constant time
        todo!("Implement constant-time verification")
    }
    
    /// Verify hash preimage in constant time
    fn verify_preimage(&self, hash_fn: &HashFunction, preimage: &[u8], expected_hash: &[u8]) -> bool {
        use sha3::{Sha3_256, Digest};
        
        let actual_hash = match hash_fn {
            HashFunction::SHA3_256 => {
                let mut hasher = Sha3_256::new();
                hasher.update(preimage);
                hasher.finalize().to_vec()
            },
            // Other hash functions...
            _ => return false,
        };
        
        // Constant-time comparison
        constant_time_eq(&actual_hash, expected_hash)
    }
    
    /// Verify commitment (e.g., Pedersen commitment)
    fn verify_commitment(&self, value: &u64, blinding: &[u8], commitment: &[u8]) -> bool {
        // C = g^value * h^blinding
        // Implement commitment verification
        todo!("Implement commitment verification")
    }
}

/// Constant-time byte array equality
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    
    let mut result = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        result |= x ^ y;
    }
    
    result == 0
}
```

### 5. Security Features

```rust
impl Drop for Witness {
    fn drop(&mut self) {
        // Explicit zeroing is handled by ZeroizeOnDrop derive
        // But we can add additional cleanup here
        eprintln!("🔒 Witness securely dropped and zeroed");
    }
}

impl Witness {
    /// Securely destroy witness after use
    pub fn destroy(mut self) {
        // Explicitly zeroize all fields
        // Drop is called automatically
    }
}

// Prevent accidental serialization
impl Witness {
    // NO Serialize derive!
    // NO to_bytes() method!
    
    /// Encrypted export ONLY (for backup/recovery)
    pub fn export_encrypted(&self, password: &str) -> Result<Vec<u8>, WitnessError> {
        // Use strong encryption (ChaCha20-Poly1305)
        // Only for explicit user backup
        todo!("Implement encrypted export")
    }
}
```

### 6. Error Handling

```rust
#[derive(Debug, thiserror::Error)]
pub enum WitnessError {
    #[error("Witness does not satisfy statement")]
    InvalidWitness,
    
    #[error("Witness type mismatch")]
    TypeMismatch,
    
    #[error("Cryptographic operation failed")]
    CryptoError,
    
    #[error("Cannot serialize witness (security violation)")]
    SerializationForbidden,
}
```

### 7. Comprehensive Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discrete_log_witness() {
        // Create witness
        let exponent = vec![42u8; 32];
        let witness = Witness::discrete_log(exponent.clone());
        
        // Create matching statement
        let (generator, public_value) = compute_discrete_log(&exponent);
        let statement = StatementBuilder::new()
            .discrete_log(generator, public_value)
            .build()
            .unwrap();
        
        // Verify witness satisfies statement
        assert!(witness.satisfies_statement(&statement));
    }

    #[test]
    fn test_preimage_witness() {
        use sha3::{Sha3_256, Digest};
        
        let preimage = b"secret message";
        let hash = Sha3_256::digest(preimage).to_vec();
        
        let witness = Witness::preimage(preimage.to_vec());
        let statement = StatementBuilder::new()
            .preimage(HashFunction::SHA3_256, hash)
            .build()
            .unwrap();
        
        assert!(witness.satisfies_statement(&statement));
    }

    #[test]
    fn test_witness_zeroization() {
        // Create witness with known data
        let secret = vec![0x42u8; 32];
        let witness = Witness::discrete_log(secret.clone());
        
        // Get pointer to secret data
        let ptr = &witness as *const Witness as *const u8;
        
        // Drop witness
        drop(witness);
        
        // Verify memory is zeroed (unsafe but for testing only)
        // In production, this is handled by zeroize crate
    }

    #[test]
    fn test_constant_time_equality() {
        let a = vec![1, 2, 3, 4];
        let b = vec![1, 2, 3, 4];
        let c = vec![1, 2, 3, 5];
        
        assert!(constant_time_eq(&a, &b));
        assert!(!constant_time_eq(&a, &c));
    }

    #[test]
    fn test_witness_type_mismatch() {
        let witness = Witness::discrete_log(vec![1, 2, 3]);
        let statement = StatementBuilder::new()
            .preimage(HashFunction::SHA3_256, vec![0; 32])
            .build()
            .unwrap();
        
        assert!(!witness.satisfies_statement(&statement));
    }
}
```

### 8. Usage Documentation

```rust
/// # Example: Creating and Using a Witness
/// 
/// ```rust
/// use nexuszero_crypto::proof::{Witness, Statement, StatementBuilder};
/// 
/// // Prover knows the secret
/// let secret_exponent = generate_random_exponent();
/// let witness = Witness::discrete_log(secret_exponent);
/// 
/// // Create public statement
/// let (g, h) = compute_public_values(&witness);
/// let statement = StatementBuilder::new()
///     .discrete_log(g, h)
///     .build()
///     .unwrap();
/// 
/// // Verify witness is valid (prover side only)
/// assert!(witness.satisfies_statement(&statement));
/// 
/// // Generate proof (next step)
/// let proof = generate_proof(&statement, &witness)?;
/// 
/// // Securely destroy witness
/// witness.destroy();
/// 
/// // Proof can be sent to verifier
/// // Witness is never transmitted!
/// ```
```

## Security Checklist
- [ ] Zeroization on drop implemented
- [ ] No serialization methods
- [ ] Constant-time operations
- [ ] Encrypted export only
- [ ] Comprehensive tests
- [ ] Documentation with security warnings

Implement complete Witness system with maximum security guarantees.
```

---

### Prompt 2.3: Proof Generation & Verification

```
Implement Proof structure and the core prove/verify algorithms for the nexuszero-crypto library.

## Conceptual Overview
A zero-knowledge proof has three components:
1. **Statement** (public) - what is being proven
2. **Witness** (secret) - the knowledge proving the statement
3. **Proof** (public) - cryptographic evidence convincing verifier

## Implementation Requirements

### 1. Proof Structure (src/proof/proof.rs)

```rust
use serde::{Deserialize, Serialize};

/// A zero-knowledge proof
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Proof {
    /// Commitment phase values
    commitments: Vec<Commitment>,
    
    /// Challenge from Fiat-Shamir transform
    challenge: Challenge,
    
    /// Response phase values
    responses: Vec<Response>,
    
    /// Proof metadata
    metadata: ProofMetadata,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Commitment {
    /// Commitment value (ring element)
    value: Vec<u8>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Challenge {
    /// Challenge value from hash
    value: [u8; 32],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Response {
    /// Response value (ring element)
    value: Vec<u8>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProofMetadata {
    /// Proof system version
    version: u8,
    
    /// Timestamp of generation
    timestamp: u64,
    
    /// Size in bytes
    size: usize,
    
    /// Security level used
    security_level: SecurityLevel,
}
```

### 2. Proof Generation (Prover Algorithm)

```rust
/// Generate a zero-knowledge proof
/// 
/// # Arguments
/// * `statement` - The public statement being proven
/// * `witness` - The secret knowledge (NOT transmitted)
/// * `params` - Cryptographic parameters
/// 
/// # Returns
/// A proof that can be publicly verified
pub fn prove(
    statement: &Statement,
    witness: &Witness,
    params: &CryptoParameters,
) -> Result<Proof, ProofError> {
    // PHASE 1: Validate inputs
    if !witness.satisfies_statement(statement) {
        return Err(ProofError::InvalidWitness);
    }
    
    // PHASE 2: Commitment Phase
    // Generate random blinding factors
    let blinding_factors = generate_blinding_factors(params)?;
    
    // Compute commitments based on statement type
    let commitments = match statement.statement_type {
        StatementType::DiscreteLog { ref generator, .. } => {
            commit_discrete_log(generator, &blinding_factors, params)?
        },
        StatementType::Preimage { ref hash_function, .. } => {
            commit_preimage(hash_function, &blinding_factors, params)?
        },
        StatementType::Range { ref min, ref max, .. } => {
            commit_range(*min, *max, &blinding_factors, params)?
        },
        _ => return Err(ProofError::UnsupportedStatementType),
    };
    
    // PHASE 3: Challenge Phase (Fiat-Shamir)
    // Hash statement + commitments to get challenge
    let challenge = compute_challenge(statement, &commitments)?;
    
    // PHASE 4: Response Phase
    // Compute responses using witness and challenge
    let responses = compute_responses(
        witness,
        &blinding_factors,
        &challenge,
        params,
    )?;
    
    // PHASE 5: Package proof
    let metadata = ProofMetadata {
        version: 1,
        timestamp: current_timestamp(),
        size: estimate_proof_size(&commitments, &responses),
        security_level: params.security_level,
    };
    
    Ok(Proof {
        commitments,
        challenge,
        responses,
        metadata,
    })
}
```

### 3. Proof Verification (Verifier Algorithm)

```rust
/// Verify a zero-knowledge proof
/// 
/// # Arguments
/// * `statement` - The public statement
/// * `proof` - The proof to verify
/// * `params` - Cryptographic parameters
/// 
/// # Returns
/// `Ok(())` if proof is valid, error otherwise
pub fn verify(
    statement: &Statement,
    proof: &Proof,
    params: &CryptoParameters,
) -> Result<(), ProofError> {
    // PHASE 1: Validate proof structure
    proof.validate()?;
    
    // PHASE 2: Recompute challenge
    let recomputed_challenge = compute_challenge(statement, &proof.commitments)?;
    
    // Verify challenge matches
    if recomputed_challenge.value != proof.challenge.value {
        return Err(ProofError::InvalidChallenge);
    }
    
    // PHASE 3: Verify responses
    // This is statement-type specific
    match statement.statement_type {
        StatementType::DiscreteLog { ref generator, ref public_value } => {
            verify_discrete_log_proof(
                generator,
                public_value,
                &proof.commitments,
                &proof.challenge,
                &proof.responses,
                params,
            )?;
        },
        StatementType::Preimage { ref hash_function, ref hash_output } => {
            verify_preimage_proof(
                hash_function,
                hash_output,
                &proof.commitments,
                &proof.challenge,
                &proof.responses,
                params,
            )?;
        },
        StatementType::Range { ref min, ref max, ref commitment } => {
            verify_range_proof(
                *min,
                *max,
                commitment,
                &proof.commitments,
                &proof.challenge,
                &proof.responses,
                params,
            )?;
        },
        _ => return Err(ProofError::UnsupportedStatementType),
    }
    
    // PHASE 4: All checks passed
    Ok(())
}
```

### 4. Fiat-Shamir Transform

```rust
/// Compute Fiat-Shamir challenge
/// 
/// Challenge = H(statement || commitments)
fn compute_challenge(
    statement: &Statement,
    commitments: &[Commitment],
) -> Result<Challenge, ProofError> {
    use sha3::{Sha3_256, Digest};
    
    let mut hasher = Sha3_256::new();
    
    // Hash statement
    hasher.update(&statement.to_bytes());
    
    // Hash all commitments
    for commitment in commitments {
        hasher.update(&commitment.value);
    }
    
    let hash_output = hasher.finalize();
    let mut challenge_bytes = [0u8; 32];
    challenge_bytes.copy_from_slice(&hash_output);
    
    Ok(Challenge {
        value: challenge_bytes,
    })
}
```

### 5. Discrete Log Proof (Example Implementation)

```rust
/// Prove knowledge of x where g^x = h (Schnorr protocol)
fn commit_discrete_log(
    generator: &[u8],
    blinding: &[Vec<u8>],
    params: &CryptoParameters,
) -> Result<Vec<Commitment>, ProofError> {
    // r ← random
    // Commitment: t = g^r
    let r = &blinding[0];
    let t = modular_exponentiation(generator, r, &params.ring_params.q)?;
    
    Ok(vec![Commitment { value: t }])
}

fn compute_responses(
    witness: &Witness,
    blinding: &[Vec<u8>],
    challenge: &Challenge,
    params: &CryptoParameters,
) -> Result<Vec<Response>, ProofError> {
    // Get secret from witness
    let x = witness.get_discrete_log_secret()?;
    let r = &blinding[0];
    let c = challenge_to_scalar(&challenge.value);
    
    // Response: s = r + c*x (mod q)
    let s = add_mod(r, &mul_mod(&c, x, params.ring_params.q), params.ring_params.q);
    
    Ok(vec![Response { value: s }])
}

fn verify_discrete_log_proof(
    generator: &[u8],
    public_value: &[u8],  // h = g^x
    commitments: &[Commitment],
    challenge: &Challenge,
    responses: &[Response],
    params: &CryptoParameters,
) -> Result<(), ProofError> {
    // Verify: g^s = t * h^c
    let g = generator;
    let h = public_value;
    let t = &commitments[0].value;
    let s = &responses[0].value;
    let c = challenge_to_scalar(&challenge.value);
    
    // Compute left side: g^s
    let left = modular_exponentiation(g, s, &params.ring_params.q)?;
    
    // Compute right side: t * h^c
    let h_to_c = modular_exponentiation(h, &c, &params.ring_params.q)?;
    let right = multiply_mod(t, &h_to_c, &params.ring_params.q)?;
    
    // Verify equality
    if left != right {
        return Err(ProofError::VerificationFailed);
    }
    
    Ok(())
}
```

### 6. Proof Serialization

```rust
impl Proof {
    /// Serialize proof to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).expect("Serialization should not fail")
    }
    
    /// Deserialize proof from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ProofError> {
        bincode::deserialize(bytes)
            .map_err(|e| ProofError::DeserializationFailed(e.to_string()))
    }
    
    /// Validate proof structure
    pub fn validate(&self) -> Result<(), ProofError> {
        if self.commitments.is_empty() {
            return Err(ProofError::InvalidStructure("No commitments".to_string()));
        }
        
        if self.responses.is_empty() {
            return Err(ProofError::InvalidStructure("No responses".to_string()));
        }
        
        if self.commitments.len() != self.responses.len() {
            return Err(ProofError::InvalidStructure("Commitment/response mismatch".to_string()));
        }
        
        Ok(())
    }
    
    /// Get proof size in bytes
    pub fn size(&self) -> usize {
        self.metadata.size
    }
}
```

### 7. Error Handling

```rust
#[derive(Debug, thiserror::Error)]
pub enum ProofError {
    #[error("Invalid witness for statement")]
    InvalidWitness,
    
    #[error("Unsupported statement type")]
    UnsupportedStatementType,
    
    #[error("Verification failed")]
    VerificationFailed,
    
    #[error("Invalid challenge")]
    InvalidChallenge,
    
    #[error("Invalid proof structure: {0}")]
    InvalidStructure(String),
    
    #[error("Cryptographic error: {0}")]
    CryptoError(String),
    
    #[error("Deserialization failed: {0}")]
    DeserializationFailed(String),
}
```

### 8. Comprehensive Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discrete_log_proof_correctness() {
        // Setup
        let params = CryptoParameters::new_128bit_security();
        let (g, x, h) = setup_discrete_log_instance();
        
        // Create statement and witness
        let statement = StatementBuilder::new()
            .discrete_log(g.clone(), h.clone())
            .build()
            .unwrap();
        let witness = Witness::discrete_log(x);
        
        // Generate proof
        let proof = prove(&statement, &witness, &params).unwrap();
        
        // Verify proof
        assert!(verify(&statement, &proof, &params).is_ok());
    }

    #[test]
    fn test_proof_with_invalid_witness() {
        let params = CryptoParameters::new_128bit_security();
        let (g, x, h) = setup_discrete_log_instance();
        
        let statement = StatementBuilder::new()
            .discrete_log(g.clone(), h.clone())
            .build()
            .unwrap();
        
        // Wrong witness
        let wrong_witness = Witness::discrete_log(vec![0; 32]);
        
        // Proof generation should fail
        assert!(prove(&statement, &wrong_witness, &params).is_err());
    }

    #[test]
    fn test_proof_tampering() {
        let params = CryptoParameters::new_128bit_security();
        let (g, x, h) = setup_discrete_log_instance();
        
        let statement = StatementBuilder::new()
            .discrete_log(g, h)
            .build()
            .unwrap();
        let witness = Witness::discrete_log(x);
        
        let mut proof = prove(&statement, &witness, &params).unwrap();
        
        // Tamper with proof
        proof.responses[0].value[0] ^= 1;
        
        // Verification should fail
        assert!(verify(&statement, &proof, &params).is_err());
    }

    #[test]
    fn test_proof_serialization() {
        let params = CryptoParameters::new_128bit_security();
        let (g, x, h) = setup_discrete_log_instance();
        
        let statement = StatementBuilder::new()
            .discrete_log(g, h)
            .build()
            .unwrap();
        let witness = Witness::discrete_log(x);
        
        let proof = prove(&statement, &witness, &params).unwrap();
        
        // Serialize and deserialize
        let bytes = proof.to_bytes();
        let recovered = Proof::from_bytes(&bytes).unwrap();
        
        // Verify recovered proof still works
        assert!(verify(&statement, &recovered, &params).is_ok());
    }

    #[test]
    fn test_fiat_shamir_consistency() {
        // Same inputs should produce same challenge
        let stmt = /* create statement */;
        let commitments = /* create commitments */;
        
        let c1 = compute_challenge(&stmt, &commitments).unwrap();
        let c2 = compute_challenge(&stmt, &commitments).unwrap();
        
        assert_eq!(c1.value, c2.value);
    }
}
```

### 9. Performance Benchmarks

```rust
#[cfg(test)]
mod benches {
    use super::*;
    use criterion::{black_box, Criterion};

    pub fn bench_proof_generation(c: &mut Criterion) {
        let params = CryptoParameters::new_128bit_security();
        let (g, x, h) = setup_discrete_log_instance();
        let statement = /* ... */;
        let witness = /* ... */;
        
        c.bench_function("prove_discrete_log", |b| {
            b.iter(|| {
                prove(black_box(&statement), black_box(&witness), black_box(&params))
            });
        });
    }

    pub fn bench_proof_verification(c: &mut Criterion) {
        let params = CryptoParameters::new_128bit_security();
        let statement = /* ... */;
        let proof = /* ... */;
        
        c.bench_function("verify_discrete_log", |b| {
            b.iter(|| {
                verify(black_box(&statement), black_box(&proof), black_box(&params))
            });
        });
    }
}
```

## Performance Targets
- Proof generation: < 100ms (128-bit security)
- Proof verification: < 50ms (128-bit security)
- Proof size: < 10KB for typical statements

Implement complete prove/verify system with all statement types.
```

---

## 🎯 DAY 5: PARAMETER SELECTION

### Prompt 3.1: Security Parameter Selection

```
Implement intelligent parameter selection algorithms for balancing security, performance, and proof size.

## Background: Security-Performance Trade-offs

Different security levels require different parameters:
- **128-bit security:** Suitable for most applications, fast verification
- **192-bit security:** High-value transactions, moderate performance
- **256-bit security:** Maximum security, slower but still practical

Parameters to optimize:
- n (lattice dimension)
- q (modulus)
- σ (error distribution)
- Affects: proof size, generation time, verification time

## Implementation Requirements

### 1. Parameter Sets (src/params/security.rs)

```rust
use serde::{Deserialize, Serialize};

/// Standard security levels (NIST post-quantum standards)
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum SecurityLevel {
    /// 128-bit security (NIST Level 1)
    /// Comparable to AES-128
    Bit128,
    
    /// 192-bit security (NIST Level 3)
    /// Comparable to AES-192
    Bit192,
    
    /// 256-bit security (NIST Level 5)
    /// Comparable to AES-256
    Bit256,
}

/// Complete parameter set for a security level
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParameterSet {
    /// Security level
    pub security_level: SecurityLevel,
    
    /// Ring dimension (power of 2)
    pub n: usize,
    
    /// Coefficient modulus (prime, q ≡ 1 mod 2n)
    pub q: u64,
    
    /// Error distribution standard deviation
    pub sigma: f64,
    
    /// Estimated proof size (bytes)
    pub proof_size: usize,
    
    /// Estimated prove time (milliseconds)
    pub prove_time_ms: u64,
    
    /// Estimated verify time (milliseconds)
    pub verify_time_ms: u64,
}

impl ParameterSet {
    /// Get standard parameter set for security level
    pub fn from_security_level(level: SecurityLevel) -> Self {
        match level {
            SecurityLevel::Bit128 => Self::standard_128bit(),
            SecurityLevel::Bit192 => Self::standard_192bit(),
            SecurityLevel::Bit256 => Self::standard_256bit(),
        }
    }
    
    /// Standard 128-bit security parameters
    fn standard_128bit() -> Self {
        ParameterSet {
            security_level: SecurityLevel::Bit128,
            n: 512,
            q: 12289,  // Prime, 12289 = 1 + 12 * 1024
            sigma: 3.2,
            proof_size: 8_192,      // ~8 KB
            prove_time_ms: 80,
            verify_time_ms: 40,
        }
    }
    
    /// Standard 192-bit security parameters
    fn standard_192bit() -> Self {
        ParameterSet {
            security_level: SecurityLevel::Bit192,
            n: 1024,
            q: 40961,  // Prime, 40961 = 1 + 20 * 2048
            sigma: 3.2,
            proof_size: 16_384,     // ~16 KB
            prove_time_ms: 150,
            verify_time_ms: 75,
        }
    }
    
    /// Standard 256-bit security parameters
    fn standard_256bit() -> Self {
        ParameterSet {
            security_level: SecurityLevel::Bit256,
            n: 2048,
            q: 65537,  // 2^16 + 1
            sigma: 3.2,
            proof_size: 32_768,     // ~32 KB
            prove_time_ms: 300,
            verify_time_ms: 150,
        }
    }
}
```

### 2. Parameter Selection Algorithm

```rust
/// Automatic parameter selection based on requirements
pub struct ParameterSelector {
    min_security: SecurityLevel,
    max_proof_size: Option<usize>,
    max_prove_time: Option<u64>,
    max_verify_time: Option<u64>,
}

impl ParameterSelector {
    pub fn new(min_security: SecurityLevel) -> Self {
        ParameterSelector {
            min_security,
            max_proof_size: None,
            max_prove_time: None,
            max_verify_time: None,
        }
    }
    
    /// Constrain proof size
    pub fn with_max_proof_size(mut self, bytes: usize) -> Self {
        self.max_proof_size = Some(bytes);
        self
    }
    
    /// Constrain prove time
    pub fn with_max_prove_time(mut self, ms: u64) -> Self {
        self.max_prove_time = Some(ms);
        self
    }
    
    /// Constrain verify time
    pub fn with_max_verify_time(mut self, ms: u64) -> Self {
        self.max_verify_time = Some(ms);
        self
    }
    
    /// Select best parameter set
    pub fn select(&self) -> Result<ParameterSet, ParameterError> {
        // Try security levels from minimum upward
        let levels = match self.min_security {
            SecurityLevel::Bit128 => vec![
                SecurityLevel::Bit128,
                SecurityLevel::Bit192,
                SecurityLevel::Bit256,
            ],
            SecurityLevel::Bit192 => vec![
                SecurityLevel::Bit192,
                SecurityLevel::Bit256,
            ],
            SecurityLevel::Bit256 => vec![
                SecurityLevel::Bit256,
            ],
        };
        
        for level in levels {
            let params = ParameterSet::from_security_level(level);
            
            // Check all constraints
            if self.satisfies_constraints(&params) {
                return Ok(params);
            }
        }
        
        Err(ParameterError::NoSuitableParameters)
    }
    
    fn satisfies_constraints(&self, params: &ParameterSet) -> bool {
        if let Some(max_size) = self.max_proof_size {
            if params.proof_size > max_size {
                return false;
            }
        }
        
        if let Some(max_time) = self.max_prove_time {
            if params.prove_time_ms > max_time {
                return false;
            }
        }
        
        if let Some(max_time) = self.max_verify_time {
            if params.verify_time_ms > max_time {
                return false;
            }
        }
        
        true
    }
}
```

### 3. Custom Parameter Generation

```rust
/// Generate custom parameters for specific requirements
pub struct CustomParameterGenerator {
    target_security: f64,  // bits of security
}

impl CustomParameterGenerator {
    pub fn new(security_bits: f64) -> Self {
        CustomParameterGenerator {
            target_security: security_bits,
        }
    }
    
    /// Generate custom parameter set
    pub fn generate(&self) -> Result<ParameterSet, ParameterError> {
        // Determine n based on security requirement
        // Rule of thumb: n ≈ security_bits * 8
        let n = self.compute_dimension();
        
        // Find suitable prime modulus
        let q = self.find_modulus(n)?;
        
        // Compute optimal sigma
        let sigma = self.compute_sigma(n, q);
        
        // Estimate performance
        let (proof_size, prove_time, verify_time) = 
            self.estimate_performance(n, q);
        
        // Determine security level classification
        let security_level = if self.target_security < 160.0 {
            SecurityLevel::Bit128
        } else if self.target_security < 224.0 {
            SecurityLevel::Bit192
        } else {
            SecurityLevel::Bit256
        };
        
        Ok(ParameterSet {
            security_level,
            n,
            q,
            sigma,
            proof_size,
            prove_time_ms: prove_time,
            verify_time_ms: verify_time,
        })
    }
    
    fn compute_dimension(&self) -> usize {
        // n should be power of 2
        let n_float = self.target_security * 8.0;
        let n = n_float.ceil() as usize;
        
        // Round up to next power of 2
        n.next_power_of_two()
    }
    
    fn find_modulus(&self, n: usize) -> Result<u64, ParameterError> {
        // Find prime q where q ≡ 1 (mod 2n)
        // This ensures primitive 2n-th root of unity exists
        
        let start = (n * 2) as u64;
        for k in 1..10000 {
            let candidate = start * k + 1;
            if self.is_prime(candidate) {
                return Ok(candidate);
            }
        }
        
        Err(ParameterError::NoSuitableModulus)
    }
    
    fn is_prime(&self, n: u64) -> bool {
        // Miller-Rabin primality test
        if n < 2 {
            return false;
        }
        if n == 2 || n == 3 {
            return true;
        }
        if n % 2 == 0 {
            return false;
        }
        
        // Implement Miller-Rabin test
        // For production, use a proper primality library
        todo!("Implement Miller-Rabin primality test")
    }
    
    fn compute_sigma(&self, n: usize, q: u64) -> f64 {
        // Sigma affects security vs correctness trade-off
        // Standard choice: sigma ≈ sqrt(n) / pi
        (n as f64).sqrt() / std::f64::consts::PI
    }
    
    fn estimate_performance(&self, n: usize, q: u64) -> (usize, u64, u64) {
        // Empirical formulas for performance estimation
        let proof_size = n * 16;  // Rough estimate: 16 bytes per dimension
        
        // Prove time scales roughly as O(n^2 log n) due to NTT
        let prove_time = ((n * n) as f64 * (n as f64).log2() / 1_000_000.0) as u64;
        
        // Verify time is similar
        let verify_time = prove_time / 2;
        
        (proof_size, prove_time, verify_time)
    }
}
```

### 4. Parameter Validation

```rust
impl ParameterSet {
    /// Validate parameter set for security and correctness
    pub fn validate(&self) -> Result<(), ParameterError> {
        // Check n is power of 2
        if !self.n.is_power_of_two() {
            return Err(ParameterError::InvalidDimension);
        }
        
        // Check q is prime (simplified check)
        if !self.is_prime_simple(self.q) {
            return Err(ParameterError::InvalidModulus);
        }
        
        // Check q ≡ 1 (mod 2n)
        if (self.q - 1) % (2 * self.n as u64) != 0 {
            return Err(ParameterError::InvalidModulus);
        }
        
        // Check sigma is reasonable
        if self.sigma < 1.0 || self.sigma > 10.0 {
            return Err(ParameterError::InvalidSigma);
        }
        
        // Verify security estimate
        let actual_security = self.estimate_security();
        let expected_security = match self.security_level {
            SecurityLevel::Bit128 => 128.0,
            SecurityLevel::Bit192 => 192.0,
            SecurityLevel::Bit256 => 256.0,
        };
        
        if actual_security < expected_security * 0.9 {
            return Err(ParameterError::InsufficientSecurity);
        }
        
        Ok(())
    }
    
    /// Estimate actual security level (simplified)
    fn estimate_security(&self) -> f64 {
        // Simplified security estimation
        // Real implementation would use lattice estimator
        
        let n_security = (self.n as f64).log2() * 16.0;
        let q_security = (self.q as f64).log2() * 4.0;
        let sigma_penalty = self.sigma.log2() * 8.0;
        
        (n_security + q_security - sigma_penalty).min(512.0)
    }
    
    fn is_prime_simple(&self, n: u64) -> bool {
        if n < 2 {
            return false;
        }
        if n == 2 {
            return true;
        }
        if n % 2 == 0 {
            return false;
        }
        
        let sqrt_n = (n as f64).sqrt() as u64;
        for i in (3..=sqrt_n).step_by(2) {
            if n % i == 0 {
                return false;
            }
        }
        true
    }
}
```

### 5. Usage Examples

```rust
#[cfg(test)]
mod examples {
    use super::*;

    #[test]
    fn example_standard_parameters() {
        // Use standard parameters for typical application
        let params = ParameterSet::from_security_level(SecurityLevel::Bit128);
        
        println!("128-bit security parameters:");
        println!("  n = {}", params.n);
        println!("  q = {}", params.q);
        println!("  sigma = {}", params.sigma);
        println!("  Proof size: {} bytes", params.proof_size);
        println!("  Prove time: {} ms", params.prove_time_ms);
        println!("  Verify time: {} ms", params.verify_time_ms);
    }

    #[test]
    fn example_constrained_selection() {
        // Select parameters with constraints
        let selector = ParameterSelector::new(SecurityLevel::Bit128)
            .with_max_proof_size(10_000)  // Max 10KB proof
            .with_max_verify_time(50);     // Max 50ms verify
        
        let params = selector.select().unwrap();
        
        assert!(params.proof_size <= 10_000);
        assert!(params.verify_time_ms <= 50);
    }

    #[test]
    fn example_custom_generation() {
        // Generate custom parameters for 160-bit security
        let generator = CustomParameterGenerator::new(160.0);
        let params = generator.generate().unwrap();
        
        println!("Custom 160-bit parameters:");
        println!("  n = {}", params.n);
        println!("  q = {}", params.q);
    }
}
```

### 6. Error Handling

```rust
#[derive(Debug, thiserror::Error)]
pub enum ParameterError {
    #[error("No suitable parameters found for constraints")]
    NoSuitableParameters,
    
    #[error("Invalid dimension (must be power of 2)")]
    InvalidDimension,
    
    #[error("Invalid modulus")]
    InvalidModulus,
    
    #[error("No suitable modulus found")]
    NoSuitableModulus,
    
    #[error("Invalid sigma parameter")]
    InvalidSigma,
    
    #[error("Insufficient security level")]
    InsufficientSecurity,
}
```

### 7. Comprehensive Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_parameter_validation() {
        let params = ParameterSet::from_security_level(SecurityLevel::Bit128);
        assert!(params.validate().is_ok());
        
        let params = ParameterSet::from_security_level(SecurityLevel::Bit192);
        assert!(params.validate().is_ok());
        
        let params = ParameterSet::from_security_level(SecurityLevel::Bit256);
        assert!(params.validate().is_ok());
    }

    #[test]
    fn test_parameter_selection() {
        let selector = ParameterSelector::new(SecurityLevel::Bit128);
        let params = selector.select().unwrap();
        
        assert_eq!(params.security_level, SecurityLevel::Bit128);
    }

    #[test]
    fn test_constrained_selection_impossible() {
        // Impossible constraints
        let selector = ParameterSelector::new(SecurityLevel::Bit256)
            .with_max_proof_size(1000)   // Too small for 256-bit
            .with_max_verify_time(10);   // Too fast for 256-bit
        
        assert!(selector.select().is_err());
    }

    #[test]
    fn test_custom_parameter_generation() {
        let generator = CustomParameterGenerator::new(128.0);
        let params = generator.generate().unwrap();
        
        assert!(params.n.is_power_of_two());
        assert!(params.validate().is_ok());
    }
}
```

## Reference: Parameter Trade-offs

| Security | n    | q     | σ   | Proof Size | Prove Time | Verify Time |
|----------|------|-------|-----|------------|------------|-------------|
| 128-bit  | 512  | 12289 | 3.2 | ~8 KB      | ~80 ms     | ~40 ms      |
| 192-bit  | 1024 | 40961 | 3.2 | ~16 KB     | ~150 ms    | ~75 ms      |
| 256-bit  | 2048 | 65537 | 3.2 | ~32 KB     | ~300 ms    | ~150 ms     |

Implement complete parameter selection system with validation and custom generation.
```

---

## ✅ DAY 6-7: UNIT TESTS WITH TEST VECTORS

### Prompt 4.1: Comprehensive Unit Tests

```
Create comprehensive unit tests for the nexuszero-crypto library covering all components.

## Test Organization

```
tests/
├── integration_tests.rs         # End-to-end integration tests
├── lattice_tests.rs              # Lattice operation tests
├── proof_tests.rs                # Proof system tests
├── parameter_tests.rs            # Parameter selection tests
└── test_vectors/
    ├── nist_vectors.json        # NIST test vectors
    ├── custom_vectors.json      # Custom test cases
    └── edge_cases.json          # Edge case tests
```

## Test Requirements

### 1. Lattice Operation Tests (tests/lattice_tests.rs)

```rust
use nexuszero_crypto::lattice::*;

#[cfg(test)]
mod lwe_tests {
    use super::*;

    #[test]
    fn test_lwe_encrypt_decrypt_correctness() {
        // Test all security levels
        for level in [SecurityLevel::Bit128, SecurityLevel::Bit192, SecurityLevel::Bit256] {
            let params = LWEParameters::from_security_level(level);
            let (sk, pk) = keygen(&params);
            
            // Test both message values
            for msg in [false, true] {
                let ct = encrypt(&pk, msg, &params);
                let recovered = decrypt(&sk, &ct, &params);
                
                assert_eq!(
                    recovered, msg,
                    "Decryption failed for {} at {:?} security",
                    msg, level
                );
            }
        }
    }

    #[test]
    fn test_lwe_encryption_randomness() {
        // Same message should produce different ciphertexts
        let params = LWEParameters::from_security_level(SecurityLevel::Bit128);
        let (_, pk) = keygen(&params);
        
        let ct1 = encrypt(&pk, true, &params);
        let ct2 = encrypt(&pk, true, &params);
        
        assert_ne!(ct1.u, ct2.u, "Ciphertexts should differ (randomized)");
    }

    #[test]
    fn test_error_distribution() {
        // Statistical test: errors should follow Gaussian distribution
        let sigma = 3.2;
        let n = 10000;
        
        let errors: Vec<i64> = (0..n)
            .map(|_| sample_error(sigma, 1)[0])
            .collect();
        
        // Compute empirical mean and standard deviation
        let mean: f64 = errors.iter().sum::<i64>() as f64 / n as f64;
        let variance: f64 = errors.iter()
            .map(|&e| {
                let diff = e as f64 - mean;
                diff * diff
            })
            .sum::<f64>() / n as f64;
        let std_dev = variance.sqrt();
        
        // Mean should be close to 0
        assert!(mean.abs() < 0.5, "Mean too far from 0: {}", mean);
        
        // Std dev should be close to sigma
        assert!(
            (std_dev - sigma).abs() < 0.5,
            "Std dev {} too far from target {}",
            std_dev, sigma
        );
    }

    #[test]
    fn test_modular_arithmetic() {
        let q = 12289;
        let a = vec![1000, 5000, 10000];
        let b = vec![2000, 6000, 11000];
        
        let sum = vector_add_mod(&a, &b, q);
        
        assert_eq!(sum[0], 3000);
        assert_eq!(sum[1], 11000);
        assert_eq!(sum[2], (21000 % q));
    }
}

#[cfg(test)]
mod ring_lwe_tests {
    use super::*;

    #[test]
    fn test_polynomial_addition() {
        let a = Polynomial::from_coeffs(vec![1, 2, 3, 4]);
        let b = Polynomial::from_coeffs(vec![5, 6, 7, 8]);
        let q = 12289;
        
        let sum = poly_add(&a, &b, q);
        
        assert_eq!(sum.coeffs, vec![6, 8, 10, 12]);
    }

    #[test]
    fn test_ntt_correctness() {
        let n = 512;
        let q = 12289;
        let root = find_primitive_root(n, q).unwrap();
        
        // Random polynomial
        let poly = Polynomial::random(n, q);
        
        // NTT then INTT should recover original
        let transformed = ntt(&poly, q, root);
        let recovered = intt(&transformed, q, root);
        
        assert_eq!(
            poly.coeffs, recovered.coeffs,
            "NTT/INTT should be inverse operations"
        );
    }

    #[test]
    fn test_ntt_multiplication_correctness() {
        let n = 512;
        let q = 12289;
        
        let a = Polynomial::random(n, q);
        let b = Polynomial::random(n, q);
        
        // Multiply using NTT
        let ntt_result = poly_mult_ntt(&a, &b, q);
        
        // Multiply using schoolbook (slow but correct)
        let schoolbook_result = poly_mult_schoolbook(&a, &b, q);
        
        assert_eq!(
            ntt_result.coeffs, schoolbook_result.coeffs,
            "NTT and schoolbook multiplication should match"
        );
    }

    #[test]
    fn test_cyclotomic_reduction() {
        // Verify X^n + 1 reduction works correctly
        let n = 512;
        let q = 12289;
        
        // Create polynomial X^n (should reduce to -1 mod X^n+1)
        let mut coeffs = vec![0i64; n + 1];
        coeffs[n] = 1;
        let poly = Polynomial::from_coeffs(coeffs);
        
        let reduced = reduce_cyclotomic(&poly, n, q);
        
        // Should be -1 ≡ q-1 (mod q)
        assert_eq!(reduced.coeffs[0], (q - 1) as i64);
        for i in 1..n {
            assert_eq!(reduced.coeffs[i], 0);
        }
    }

    #[test]
    fn test_ring_lwe_encryption_decryption() {
        let params = RingLWEParameters::new_128bit_security();
        let (sk, pk) = ring_keygen(&params);
        
        // Test with random message bits
        let message = vec![true, false, true, true, false];
        let ct = ring_encrypt(&pk, &message, &params);
        let recovered = ring_decrypt(&sk, &ct, &params);
        
        assert_eq!(message, recovered, "Ring-LWE decryption failed");
    }
}
```

### 2. Proof System Tests (tests/proof_tests.rs)

```rust
#[cfg(test)]
mod proof_correctness_tests {
    use super::*;

    #[test]
    fn test_discrete_log_proof() {
        let params = CryptoParameters::new_128bit_security();
        
        // Generate discrete log instance
        let (g, x, h) = generate_discrete_log_instance();
        
        // Create statement and witness
        let statement = StatementBuilder::new()
            .discrete_log(g, h)
            .build()
            .unwrap();
        
        let witness = Witness::discrete_log(x);
        
        // Generate and verify proof
        let proof = prove(&statement, &witness, &params).unwrap();
        assert!(verify(&statement, &proof, &params).is_ok());
    }

    #[test]
    fn test_preimage_proof() {
        use sha3::{Sha3_256, Digest};
        
        let params = CryptoParameters::new_128bit_security();
        let preimage = b"secret message";
        let hash = Sha3_256::digest(preimage).to_vec();
        
        let statement = StatementBuilder::new()
            .preimage(HashFunction::SHA3_256, hash)
            .build()
            .unwrap();
        
        let witness = Witness::preimage(preimage.to_vec());
        
        let proof = prove(&statement, &witness, &params).unwrap();
        assert!(verify(&statement, &proof, &params).is_ok());
    }

    #[test]
    fn test_range_proof() {
        let params = CryptoParameters::new_128bit_security();
        let value = 42u64;
        let min = 0u64;
        let max = 100u64;
        
        // Generate commitment
        let (commitment, blinding) = commit_to_value(value);
        
        let statement = StatementBuilder::new()
            .range(min, max, commitment)
            .build()
            .unwrap();
        
        let witness = Witness::range(value, blinding);
        
        let proof = prove(&statement, &witness, &params).unwrap();
        assert!(verify(&statement, &proof, &params).is_ok());
    }

    #[test]
    fn test_proof_soundness() {
        // Attacker tries to prove false statement
        let params = CryptoParameters::new_128bit_security();
        let (g, x, h) = generate_discrete_log_instance();
        
        // Correct statement
        let statement = StatementBuilder::new()
            .discrete_log(g.clone(), h)
            .build()
            .unwrap();
        
        // WRONG witness (different exponent)
        let wrong_x = generate_random_exponent();
        let wrong_witness = Witness::discrete_log(wrong_x);
        
        // Should fail to generate valid proof
        let result = prove(&statement, &wrong_witness, &params);
        assert!(result.is_err(), "Should reject invalid witness");
    }

    #[test]
    fn test_proof_zero_knowledge() {
        // Verifier learns nothing from proof except validity
        // This is hard to test directly, but we can verify:
        // 1. Proof doesn't contain witness
        // 2. Proof size is independent of witness
        
        let params = CryptoParameters::new_128bit_security();
        
        // Two different witnesses
        let (g, x1, h1) = generate_discrete_log_instance();
        let (_, x2, h2) = generate_discrete_log_instance();
        
        let stmt1 = StatementBuilder::new().discrete_log(g.clone(), h1).build().unwrap();
        let stmt2 = StatementBuilder::new().discrete_log(g, h2).build().unwrap();
        
        let witness1 = Witness::discrete_log(x1);
        let witness2 = Witness::discrete_log(x2);
        
        let proof1 = prove(&stmt1, &witness1, &params).unwrap();
        let proof2 = prove(&stmt2, &witness2, &params).unwrap();
        
        // Proofs should have similar size
        let size_diff = (proof1.size() as i64 - proof2.size() as i64).abs();
        assert!(
            size_diff < 100,
            "Proof sizes should be similar (zero-knowledge property)"
        );
    }
}

#[cfg(test)]
mod proof_security_tests {
    use super::*;

    #[test]
    fn test_proof_tampering_detection() {
        let params = CryptoParameters::new_128bit_security();
        let (g, x, h) = generate_discrete_log_instance();
        
        let statement = StatementBuilder::new()
            .discrete_log(g, h)
            .build()
            .unwrap();
        let witness = Witness::discrete_log(x);
        
        let mut proof = prove(&statement, &witness, &params).unwrap();
        
        // Tamper with different parts
        let tamper_tests = vec![
            ("commitment", || proof.commitments[0].value[0] ^= 1),
            ("challenge", || proof.challenge.value[0] ^= 1),
            ("response", || proof.responses[0].value[0] ^= 1),
        ];
        
        for (part, tamper_fn) in tamper_tests {
            let original_proof = proof.clone();
            tamper_fn();
            
            let result = verify(&statement, &proof, &params);
            assert!(
                result.is_err(),
                "Tampering {} should be detected",
                part
            );
            
            // Restore for next test
            proof = original_proof;
        }
    }

    #[test]
    fn test_replay_attack_resistance() {
        // Same proof shouldn't verify for different statement
        let params = CryptoParameters::new_128bit_security();
        
        let (g, x1, h1) = generate_discrete_log_instance();
        let (_, x2, h2) = generate_discrete_log_instance();
        
        let stmt1 = StatementBuilder::new().discrete_log(g.clone(), h1).build().unwrap();
        let stmt2 = StatementBuilder::new().discrete_log(g, h2).build().unwrap();
        
        let witness1 = Witness::discrete_log(x1);
        let proof1 = prove(&stmt1, &witness1, &params).unwrap();
        
        // Try to use proof1 for stmt2 (replay attack)
        let result = verify(&stmt2, &proof1, &params);
        assert!(
            result.is_err(),
            "Proof for one statement shouldn't verify for another"
        );
    }
}
```

### 3. Test Vectors (tests/test_vectors/nist_vectors.json)

```json
{
  "test_vectors": [
    {
      "id": "LWE-128-001",
      "security_level": "128-bit",
      "parameters": {
        "n": 512,
        "q": 12289,
        "sigma": 3.2
      },
      "secret_key": "0x1a2b3c...",
      "public_key_A": "0x4d5e6f...",
      "public_key_b": "0x7g8h9i...",
      "message": false,
      "ciphertext_u": "0xaabbcc...",
      "ciphertext_v": 5432,
      "decryption_result": false
    },
    {
      "id": "PROOF-DL-001",
      "statement_type": "DiscreteLog",
      "generator": "0x123456...",
      "public_value": "0x789abc...",
      "witness_exponent": "0xdef012...",
      "proof": {
        "commitment": "0x345678...",
        "challenge": "0x9abcde...",
        "response": "0xf01234..."
      },
      "verification_result": true
    }
  ]
}
```

### 4. Test Vector Execution

```rust
use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Debug, Deserialize)]
struct TestVectorSet {
    test_vectors: Vec<TestVector>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "id")]
enum TestVector {
    #[serde(rename = "LWE-128-001")]
    LWEEncryption {
        security_level: String,
        parameters: LWETestParams,
        secret_key: String,
        public_key_A: String,
        public_key_b: String,
        message: bool,
        ciphertext_u: String,
        ciphertext_v: u64,
        decryption_result: bool,
    },
    ProofDiscreteLog {
        statement_type: String,
        generator: String,
        public_value: String,
        witness_exponent: String,
        proof: ProofTestData,
        verification_result: bool,
    },
}

#[test]
fn test_nist_vectors() {
    let vectors_json = fs::read_to_string("tests/test_vectors/nist_vectors.json")
        .expect("Failed to read NIST test vectors");
    
    let test_set: TestVectorSet = serde_json::from_str(&vectors_json)
        .expect("Failed to parse test vectors");
    
    for vector in test_set.test_vectors {
        match vector {
            TestVector::LWEEncryption { .. } => test_lwe_vector(vector),
            TestVector::ProofDiscreteLog { .. } => test_proof_vector(vector),
        }
    }
}
```

### 5. Property-Based Testing

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn prop_lwe_correctness(
        message in prop::bool::ANY,
        security_level in 0u8..3,
    ) {
        let level = match security_level {
            0 => SecurityLevel::Bit128,
            1 => SecurityLevel::Bit192,
            _ => SecurityLevel::Bit256,
        };
        
        let params = LWEParameters::from_security_level(level);
        let (sk, pk) = keygen(&params);
        
        let ct = encrypt(&pk, message, &params);
        let recovered = decrypt(&sk, &ct, &params);
        
        prop_assert_eq!(recovered, message);
    }

    #[test]
    fn prop_proof_verification(
        exponent in prop::collection::vec(any::<u8>(), 32),
    ) {
        let params = CryptoParameters::new_128bit_security();
        let (g, h) = compute_discrete_log(&exponent);
        
        let statement = StatementBuilder::new()
            .discrete_log(g, h)
            .build()
            .unwrap();
        
        let witness = Witness::discrete_log(exponent);
        let proof = prove(&statement, &witness, &params).unwrap();
        
        prop_assert!(verify(&statement, &proof, &params).is_ok());
    }
}
```

### 6. Benchmark Tests

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_lwe_encryption(c: &mut Criterion) {
    let params = LWEParameters::from_security_level(SecurityLevel::Bit128);
    let (_, pk) = keygen(&params);
    
    c.bench_function("lwe_encrypt_128bit", |b| {
        b.iter(|| {
            encrypt(black_box(&pk), black_box(true), black_box(&params))
        });
    });
}

fn bench_proof_generation(c: &mut Criterion) {
    let params = CryptoParameters::new_128bit_security();
    let (g, x, h) = generate_discrete_log_instance();
    let statement = StatementBuilder::new().discrete_log(g, h).build().unwrap();
    let witness = Witness::discrete_log(x);
    
    c.bench_function("prove_discrete_log_128bit", |b| {
        b.iter(|| {
            prove(
                black_box(&statement),
                black_box(&witness),
                black_box(&params)
            )
        });
    });
}

fn bench_proof_verification(c: &mut Criterion) {
    let params = CryptoParameters::new_128bit_security();
    let (g, x, h) = generate_discrete_log_instance();
    let statement = StatementBuilder::new().discrete_log(g, h).build().unwrap();
    let witness = Witness::discrete_log(x);
    let proof = prove(&statement, &witness, &params).unwrap();
    
    c.bench_function("verify_discrete_log_128bit", |b| {
        b.iter(|| {
            verify(
                black_box(&statement),
                black_box(&proof),
                black_box(&params)
            )
        });
    });
}

criterion_group!(
    benches,
    bench_lwe_encryption,
    bench_proof_generation,
    bench_proof_verification
);
criterion_main!(benches);
```

## Test Coverage Goals
- Line coverage: >90%
- Branch coverage: >85%
- All public APIs tested
- Edge cases covered
- Security properties verified

Run all tests with:
```bash
cargo test --all-features
cargo test --release  # Performance tests
cargo bench           # Benchmarks
```

Implement comprehensive test suite with NIST vectors and property-based testing.
```

---

## 📚 WEEK 1 SUMMARY

### Deliverables Checklist

- [ ] Rust project structure created
- [ ] LWE primitives implemented and tested
- [ ] Ring-LWE with NTT optimization implemented
- [ ] Statement structure with builder pattern
- [ ] Witness structure with security guarantees
- [ ] Proof generation and verification algorithms
- [ ] Parameter selection with validation
- [ ] Comprehensive unit tests (>90% coverage)
- [ ] NIST test vectors passing
- [ ] Benchmarks showing performance targets met

### Performance Targets (128-bit Security)

| Operation | Target | Acceptance Threshold |
|-----------|--------|---------------------|
| LWE Encryption | < 5ms | < 10ms |
| Ring-LWE Encryption | < 2ms | < 5ms |
| Proof Generation | < 100ms | < 150ms |
| Proof Verification | < 50ms | < 75ms |
| Proof Size | < 10KB | < 15KB |

### Next Week Preview

Week 2 will focus on the Neural Optimizer - using PyTorch to build a GNN that optimizes proof parameters in real-time.

---

**Created:** November 20, 2024  
**Purpose:** Complete Copilot prompt set for Week 1 Cryptography Module  
**Status:** Ready for execution in VS Code
