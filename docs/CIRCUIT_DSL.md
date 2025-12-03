# NexusZero Circuit DSL & Constraint System

This document provides comprehensive documentation for the NexusZero circuit definition language (DSL) and constraint system for building custom zero-knowledge proof circuits.

## Table of Contents

1. [Overview](#overview)
2. [Core Concepts](#core-concepts)
3. [Circuit Components](#circuit-components)
4. [Constraint System](#constraint-system)
5. [Witness DSL](#witness-dsl)
6. [Circuit Composition](#circuit-composition)
7. [Built-in Gadgets](#built-in-gadgets)
8. [Advanced Patterns](#advanced-patterns)
9. [Examples](#examples)
10. [Best Practices](#best-practices)

---

## Overview

The NexusZero Circuit DSL provides a high-level abstraction for defining zero-knowledge proof circuits. It allows you to:

- Define custom constraints and relationships
- Compose circuits from reusable components
- Generate witnesses programmatically
- Support multiple proof backends (Groth16, PLONK, Bulletproofs)

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          CIRCUIT DSL ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                   │
│  │   Circuit   │────▶│  Compiler   │────▶│  Backend    │                   │
│  │    DSL      │     │             │     │ (Groth16,   │                   │
│  │             │     │             │     │  PLONK...)  │                   │
│  └─────────────┘     └─────────────┘     └─────────────┘                   │
│        │                    │                   │                          │
│        ▼                    ▼                   ▼                          │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                   │
│  │ Components  │     │ Constraints │     │   Proof     │                   │
│  │ & Variables │     │  (R1CS/     │     │             │                   │
│  │             │     │  PLONK)     │     │             │                   │
│  └─────────────┘     └─────────────┘     └─────────────┘                   │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  LAYERS: High-Level DSL → IR (Intermediate) → Backend-Specific → Proof     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Concepts

### Variables

Variables represent values in a circuit. They can be:

- **Public inputs** - Known to both prover and verifier
- **Private inputs** - Known only to the prover (witness)
- **Intermediate** - Computed during circuit execution

```rust
use nexuszero_crypto::proof::circuit::{Variable, VariableType};

// Define a variable
let age = Variable {
    name: "age".to_string(),
    var_type: VariableType::Integer { signed: false },
    bit_length: 8,
};

// Variable types
pub enum VariableType {
    FieldElement,           // Native field element
    Boolean,                // 0 or 1
    Integer { signed: bool }, // Integer with optional sign
    Bytes { length: usize },  // Fixed-length byte array
}
```

### Constraints

Constraints define relationships between variables that must hold for a valid proof.

```rust
use nexuszero_crypto::proof::circuit::{Constraint, Expression, ConstraintType};

// Equality constraint: a + b = c
let constraint = Constraint {
    left: Expression::Add(
        Box::new(Expression::Variable("a".to_string())),
        Box::new(Expression::Variable("b".to_string()))
    ),
    right: Expression::Variable("c".to_string()),
    constraint_type: ConstraintType::Equality,
};

// Expression types
pub enum Expression {
    Variable(String),
    Constant(Vec<u8>),
    Add(Box<Expression>, Box<Expression>),
    Mul(Box<Expression>, Box<Expression>),
    // Additional operations...
}

// Constraint types
pub enum ConstraintType {
    Equality,                    // left == right
    Range { min: u64, max: u64 }, // min <= value <= max
    Boolean,                     // value ∈ {0, 1}
    Custom(String),              // Custom constraint logic
}
```

### Circuits

A Circuit combines variables, constraints, and execution flow.

```rust
use nexuszero_crypto::proof::circuit::{Circuit, Connection};

let circuit = Circuit {
    id: "age_verification".to_string(),
    components: vec!["range_check".to_string()],
    connections: vec![
        Connection {
            from_component: "input".to_string(),
            from_output: "age".to_string(),
            to_component: "range_check".to_string(),
            to_input: "value".to_string(),
        }
    ],
    public_inputs: vec!["minimum_age".to_string()],
    private_inputs: vec!["actual_age".to_string()],
    outputs: vec!["is_valid".to_string()],
};
```

---

## Circuit Components

### CircuitComponent Trait

Every reusable circuit piece implements this trait:

```rust
#[async_trait]
pub trait CircuitComponent: Send + Sync {
    /// Unique identifier for this component
    fn id(&self) -> &str;

    /// Input variables required
    fn inputs(&self) -> Vec<Variable>;

    /// Output variables produced
    fn outputs(&self) -> Vec<Variable>;

    /// Constraints imposed by this component
    fn constraints(&self) -> Vec<Constraint>;

    /// Generate witness data for this component
    async fn generate_witness(
        &self,
        inputs: &HashMap<String, Vec<u8>>
    ) -> CryptoResult<WitnessData>;
}
```

### Example: Range Check Component

```rust
use nexuszero_crypto::proof::circuit::*;
use std::collections::HashMap;

/// Component that verifies a value is within a range
pub struct RangeCheckComponent {
    min: u64,
    max: u64,
    bit_length: usize,
}

impl RangeCheckComponent {
    pub fn new(min: u64, max: u64) -> Self {
        let bit_length = ((max - min) as f64).log2().ceil() as usize + 1;
        Self { min, max, bit_length }
    }
}

#[async_trait::async_trait]
impl CircuitComponent for RangeCheckComponent {
    fn id(&self) -> &str {
        "range_check"
    }

    fn inputs(&self) -> Vec<Variable> {
        vec![
            Variable {
                name: "value".to_string(),
                var_type: VariableType::Integer { signed: false },
                bit_length: 64,
            },
        ]
    }

    fn outputs(&self) -> Vec<Variable> {
        vec![
            Variable {
                name: "in_range".to_string(),
                var_type: VariableType::Boolean,
                bit_length: 1,
            },
        ]
    }

    fn constraints(&self) -> Vec<Constraint> {
        vec![
            // value >= min
            Constraint {
                left: Expression::Variable("value".to_string()),
                right: Expression::Constant(self.min.to_be_bytes().to_vec()),
                constraint_type: ConstraintType::Range {
                    min: self.min,
                    max: self.max
                },
            },
        ]
    }

    async fn generate_witness(
        &self,
        inputs: &HashMap<String, Vec<u8>>
    ) -> CryptoResult<WitnessData> {
        let value_bytes = inputs.get("value")
            .ok_or(CryptoError::InvalidParameter("Missing value".into()))?;

        let value = u64::from_be_bytes(
            value_bytes[..8].try_into()
                .map_err(|_| CryptoError::InvalidInput("Invalid value bytes".into()))?
        );

        let in_range = value >= self.min && value <= self.max;

        let mut variables = HashMap::new();
        variables.insert("value".to_string(), value_bytes.clone());
        variables.insert("in_range".to_string(), vec![in_range as u8]);

        Ok(WitnessData {
            variables,
            randomness: vec![0u8; 32], // Generate proper randomness in production
        })
    }
}
```

---

## Constraint System

### R1CS Constraints

The underlying constraint system uses Rank-1 Constraint System (R1CS):

```
A · B = C
```

Where A, B, C are linear combinations of variables.

```rust
// R1CS constraint: (a + b) * 1 = c
// A = a + b
// B = 1
// C = c

// In DSL form:
let addition_constraint = Constraint {
    left: Expression::Add(
        Box::new(Expression::Variable("a".into())),
        Box::new(Expression::Variable("b".into()))
    ),
    right: Expression::Variable("c".into()),
    constraint_type: ConstraintType::Equality,
};
```

### Boolean Constraints

Force a variable to be 0 or 1:

```rust
// b * (1 - b) = 0  ⟹  b ∈ {0, 1}
let boolean_constraint = Constraint {
    left: Expression::Mul(
        Box::new(Expression::Variable("b".into())),
        Box::new(Expression::Add(
            Box::new(Expression::Constant(vec![1])),
            Box::new(Expression::Mul(
                Box::new(Expression::Constant(vec![255])), // -1 in field
                Box::new(Expression::Variable("b".into()))
            ))
        ))
    ),
    right: Expression::Constant(vec![0]),
    constraint_type: ConstraintType::Equality,
};
```

### Range Constraints

Prove a value is within bounds using bit decomposition:

```rust
/// Decompose value into bits and verify each bit is boolean
fn range_constraints(value: &str, bits: usize) -> Vec<Constraint> {
    let mut constraints = Vec::new();

    // Each bit must be boolean
    for i in 0..bits {
        let bit_name = format!("{}_{}", value, i);
        constraints.push(Constraint {
            left: Expression::Variable(bit_name.clone()),
            right: Expression::Variable(bit_name.clone()),
            constraint_type: ConstraintType::Boolean,
        });
    }

    // Sum of bits * powers of 2 = value
    // Σ(bit_i * 2^i) = value

    constraints
}
```

---

## Witness DSL

The Witness DSL provides declarative witness generation.

### Witness Strategies

```rust
use nexuszero_crypto::proof::witness_dsl::*;

/// Direct strategy: transform input directly
let direct_strategy = WitnessStrategy::Direct(DirectStrategy {
    transformation: DataTransformation::Hash {
        algorithm: HashAlgorithm::SHA3_256
    },
});

/// Derived strategy: compute from other witnesses
let derived_strategy = WitnessStrategy::Derived(DerivedStrategy {
    sources: vec!["witness_a".into(), "witness_b".into()],
    operation: DerivationOperation::Combine {
        combiner: Combiner::HashCombine {
            algorithm: HashAlgorithm::Blake3
        },
    },
});

/// Circuit strategy: use circuit to generate witness
let circuit_strategy = WitnessStrategy::Circuit(CircuitStrategy {
    circuit_id: "witness_gen_circuit".into(),
    input_mapping: [
        ("secret".into(), "circuit_input".into())
    ].into(),
});
```

### Data Transformations

```rust
pub enum DataTransformation {
    /// Pass through unchanged
    Identity,

    /// Apply hash function
    Hash { algorithm: HashAlgorithm },

    /// Modular exponentiation: base^data mod modulus
    ModularExp {
        base: Vec<u8>,
        modulus: Vec<u8>
    },

    /// Pedersen commitment
    PedersenCommit {
        generator_g: Vec<u8>,
        generator_h: Vec<u8>
    },

    /// Custom transformation
    Custom {
        function_name: String,
        parameters: HashMap<String, serde_json::Value>
    },
}
```

### Witness Generation Plan

```rust
use nexuszero_crypto::proof::witness_dsl::*;

// Define a complete witness generation plan
let plan = WitnessGenerationPlan {
    statement_type: StatementType::Preimage {
        hash_function: HashFunction::SHA3_256,
        hash_output: vec![0xAB; 32],
    },
    strategy: WitnessStrategy::Direct(DirectStrategy {
        transformation: DataTransformation::Identity,
    }),
    validation_rules: vec![
        ValidationRule::SatisfiesStatement,
        ValidationRule::Size { expected_bytes: 32 },
    ],
};

// Build witness generator with multiple plans
let generator = WitnessBuilder::new()
    .add_plan(
        StatementType::DiscreteLog {
            generator: vec![],
            public_value: vec![]
        },
        discrete_log_plan
    )
    .add_plan(
        StatementType::Range {
            min: 0,
            max: 0,
            commitment: vec![]
        },
        range_plan
    )
    .build();

// Generate witness for a statement
let inputs = HashMap::from([
    ("secret".to_string(), secret_data.to_vec()),
]);
let witness = generator.generate_witness(&statement, inputs).await?;
```

---

## Circuit Composition

### Composing Multiple Components

```rust
use nexuszero_crypto::proof::circuit::*;

// 1. Define components
struct HashComponent;
struct CompareComponent;

// 2. Create circuit that combines them
let circuit = Circuit {
    id: "hash_and_compare".to_string(),
    components: vec![
        "hash".to_string(),
        "compare".to_string(),
    ],
    connections: vec![
        // Connect hash output to compare input
        Connection {
            from_component: "hash".into(),
            from_output: "digest".into(),
            to_component: "compare".into(),
            to_input: "computed_hash".into(),
        },
    ],
    public_inputs: vec!["expected_hash".into()],
    private_inputs: vec!["preimage".into()],
    outputs: vec!["is_equal".into()],
};

// 3. Register and execute
let mut engine = CircuitEngine::new();
engine.register_component(Box::new(HashComponent));
engine.register_component(Box::new(CompareComponent));
engine.register_circuit(circuit);

let inputs = HashMap::from([
    ("preimage".into(), preimage.to_vec()),
    ("expected_hash".into(), expected.to_vec()),
]);

let proof = engine.prove("hash_and_compare", inputs).await?;
```

### Nested Circuits

```rust
// Inner circuit: Range check
let inner_circuit = Circuit {
    id: "range_check".into(),
    components: vec!["decompose".into(), "verify_bits".into()],
    // ...
    outputs: vec!["in_range".into()],
};

// Outer circuit: Uses range check as component
let outer_circuit = Circuit {
    id: "balance_proof".into(),
    components: vec![
        "range_check".into(),  // Inner circuit as component
        "commitment".into(),
    ],
    connections: vec![
        Connection {
            from_component: "input".into(),
            from_output: "balance".into(),
            to_component: "range_check".into(),
            to_input: "value".into(),
        },
    ],
    // ...
};
```

---

## Built-in Gadgets

### Hash Gadgets

```rust
// SHA256 gadget
pub struct SHA256Gadget;

impl CircuitComponent for SHA256Gadget {
    fn id(&self) -> &str { "sha256" }

    fn inputs(&self) -> Vec<Variable> {
        vec![Variable {
            name: "preimage".into(),
            var_type: VariableType::Bytes { length: 64 },
            bit_length: 512,
        }]
    }

    fn outputs(&self) -> Vec<Variable> {
        vec![Variable {
            name: "digest".into(),
            var_type: VariableType::Bytes { length: 32 },
            bit_length: 256,
        }]
    }

    fn constraints(&self) -> Vec<Constraint> {
        // SHA256 round constraints (~20,000 constraints)
        sha256_constraints()
    }
}

// BLAKE3 gadget (more efficient)
pub struct BLAKE3Gadget;

// Poseidon gadget (ZK-friendly)
pub struct PoseidonGadget {
    width: usize,
    rounds: usize,
}
```

### Arithmetic Gadgets

```rust
// Addition with overflow check
pub struct SafeAddGadget;

// Multiplication
pub struct MulGadget;

// Division with quotient and remainder
pub struct DivModGadget;

// Comparison (less than, greater than, equals)
pub struct CompareGadget;
```

### Cryptographic Gadgets

```rust
// EdDSA signature verification
pub struct EdDSAVerifyGadget;

// ECDSA signature verification
pub struct ECDSAVerifyGadget;

// Merkle tree inclusion proof
pub struct MerkleProofGadget {
    depth: usize,
}

// Pedersen commitment
pub struct PedersenGadget;
```

---

## Advanced Patterns

### Recursive Proofs

Verify one proof inside another circuit:

```rust
// Circuit that verifies another proof
pub struct RecursiveVerifierCircuit {
    inner_verification_key: VerificationKey,
}

impl CircuitComponent for RecursiveVerifierCircuit {
    fn id(&self) -> &str { "recursive_verifier" }

    fn inputs(&self) -> Vec<Variable> {
        vec![
            Variable {
                name: "inner_proof".into(),
                var_type: VariableType::Bytes { length: 256 },
                bit_length: 2048,
            },
            Variable {
                name: "inner_public_inputs".into(),
                var_type: VariableType::Bytes { length: 64 },
                bit_length: 512,
            },
        ]
    }

    fn constraints(&self) -> Vec<Constraint> {
        // Groth16/PLONK verification constraints
        pairing_constraints(&self.inner_verification_key)
    }
}
```

### Lookup Tables

Efficient constraints using precomputed tables:

```rust
pub struct LookupTableGadget {
    table: Vec<(Vec<u8>, Vec<u8>)>, // (input, output) pairs
}

impl LookupTableGadget {
    // For example: S-box for AES
    pub fn aes_sbox() -> Self {
        Self {
            table: AES_SBOX.iter()
                .enumerate()
                .map(|(i, &o)| (vec![i as u8], vec![o]))
                .collect()
        }
    }

    // Generate lookup constraints
    fn lookup_constraints(&self) -> Vec<Constraint> {
        // PLONK-style lookup argument
        // ...
    }
}
```

### Custom Gates (PLONK)

```rust
// Custom gate for repeated operations
pub struct CustomGate {
    selector: String,
    wire_coefficients: [i64; 5], // q_L, q_R, q_O, q_M, q_C
}

impl CustomGate {
    /// Addition gate: q_L * a + q_R * b - q_O * c = 0
    pub fn addition() -> Self {
        Self {
            selector: "add".into(),
            wire_coefficients: [1, 1, -1, 0, 0],
        }
    }

    /// Multiplication gate: q_M * a * b - q_O * c = 0
    pub fn multiplication() -> Self {
        Self {
            selector: "mul".into(),
            wire_coefficients: [0, 0, -1, 1, 0],
        }
    }

    /// Boolean gate: a * (1 - a) = 0
    pub fn boolean() -> Self {
        Self {
            selector: "bool".into(),
            wire_coefficients: [1, 0, 0, -1, 0],
        }
    }
}
```

---

## Examples

### Example 1: Preimage Proof Circuit

```rust
use nexuszero_crypto::proof::*;
use nexuszero_crypto::proof::circuit::*;

/// Circuit proving knowledge of hash preimage
pub async fn preimage_circuit_example() -> CryptoResult<()> {
    // 1. Define the hash component
    struct SHA3Component;

    #[async_trait::async_trait]
    impl CircuitComponent for SHA3Component {
        fn id(&self) -> &str { "sha3_256" }

        fn inputs(&self) -> Vec<Variable> {
            vec![Variable {
                name: "preimage".into(),
                var_type: VariableType::Bytes { length: 32 },
                bit_length: 256,
            }]
        }

        fn outputs(&self) -> Vec<Variable> {
            vec![Variable {
                name: "hash".into(),
                var_type: VariableType::Bytes { length: 32 },
                bit_length: 256,
            }]
        }

        fn constraints(&self) -> Vec<Constraint> {
            // SHA3-256 internal constraints
            vec![]
        }

        async fn generate_witness(
            &self,
            inputs: &HashMap<String, Vec<u8>>
        ) -> CryptoResult<WitnessData> {
            use sha3::{Digest, Sha3_256};

            let preimage = inputs.get("preimage")
                .ok_or(CryptoError::InvalidParameter("Missing preimage".into()))?;

            let mut hasher = Sha3_256::new();
            hasher.update(preimage);
            let hash = hasher.finalize().to_vec();

            let mut vars = HashMap::new();
            vars.insert("preimage".into(), preimage.clone());
            vars.insert("hash".into(), hash);

            Ok(WitnessData {
                variables: vars,
                randomness: vec![0u8; 32],
            })
        }
    }

    // 2. Create circuit
    let circuit = Circuit {
        id: "preimage_proof".into(),
        components: vec!["sha3_256".into()],
        connections: vec![],
        public_inputs: vec!["expected_hash".into()],
        private_inputs: vec!["preimage".into()],
        outputs: vec!["hash".into()],
    };

    // 3. Setup engine
    let mut engine = CircuitEngine::new();
    engine.register_component(Box::new(SHA3Component));
    engine.register_circuit(circuit);

    // 4. Generate proof
    let secret = b"my_secret_preimage";
    let inputs = HashMap::from([
        ("preimage".into(), secret.to_vec()),
    ]);

    let proof = engine.prove("preimage_proof", inputs).await?;

    println!("Preimage proof generated!");
    Ok(())
}
```

### Example 2: Range Proof with Witness DSL

```rust
use nexuszero_crypto::proof::witness_dsl::*;

/// Generate range proof using witness DSL
pub async fn range_proof_dsl_example() -> CryptoResult<()> {
    // 1. Define witness generation plan
    let range_plan = WitnessGenerationPlan {
        statement_type: StatementType::Range {
            min: 0,
            max: 1000000,
            commitment: vec![],
        },
        strategy: WitnessStrategy::Direct(DirectStrategy {
            transformation: DataTransformation::PedersenCommit {
                generator_g: BULLETPROOF_G.to_vec(),
                generator_h: BULLETPROOF_H.to_vec(),
            },
        }),
        validation_rules: vec![
            ValidationRule::SatisfiesStatement,
        ],
    };

    // 2. Build generator
    let generator = WitnessBuilder::new()
        .add_plan(
            StatementType::Range { min: 0, max: 0, commitment: vec![] },
            range_plan
        )
        .build();

    // 3. Create statement with commitment
    let value = 42u64;
    let blinding = vec![0u8; 32]; // Use proper randomness!
    let commitment = bulletproofs::pedersen_commit(value, &blinding)?;

    let statement = StatementBuilder::new()
        .range(0, 1000000, commitment)
        .build()?;

    // 4. Generate witness
    let inputs = HashMap::from([
        ("value".into(), value.to_be_bytes().to_vec()),
        ("blinding".into(), blinding),
    ]);

    let witness = generator.generate_witness(&statement, inputs).await?;

    // 5. Generate proof
    let proof = prove(&statement, &witness)?;

    // 6. Verify
    verify(&statement, &proof)?;

    println!("Range proof verified! Value {} is in [0, 1000000]", value);
    Ok(())
}
```

### Example 3: Merkle Tree Membership

```rust
/// Prove membership in a Merkle tree without revealing position
pub async fn merkle_membership_example() -> CryptoResult<()> {
    // Merkle tree component
    struct MerklePathComponent {
        depth: usize,
    }

    #[async_trait::async_trait]
    impl CircuitComponent for MerklePathComponent {
        fn id(&self) -> &str { "merkle_path" }

        fn inputs(&self) -> Vec<Variable> {
            let mut inputs = vec![
                Variable {
                    name: "leaf".into(),
                    var_type: VariableType::Bytes { length: 32 },
                    bit_length: 256,
                },
                Variable {
                    name: "root".into(),
                    var_type: VariableType::Bytes { length: 32 },
                    bit_length: 256,
                },
            ];

            // Path elements and directions
            for i in 0..self.depth {
                inputs.push(Variable {
                    name: format!("sibling_{}", i),
                    var_type: VariableType::Bytes { length: 32 },
                    bit_length: 256,
                });
                inputs.push(Variable {
                    name: format!("direction_{}", i),
                    var_type: VariableType::Boolean,
                    bit_length: 1,
                });
            }

            inputs
        }

        fn constraints(&self) -> Vec<Constraint> {
            // For each level:
            // - Select left/right based on direction bit
            // - Hash siblings to get parent
            // - Final hash must equal root
            vec![]
        }

        async fn generate_witness(
            &self,
            inputs: &HashMap<String, Vec<u8>>
        ) -> CryptoResult<WitnessData> {
            // Compute intermediate hashes along path
            // ...
            Ok(WitnessData {
                variables: HashMap::new(),
                randomness: vec![],
            })
        }
    }

    // Usage
    let component = MerklePathComponent { depth: 20 };

    let mut engine = CircuitEngine::new();
    engine.register_component(Box::new(component));

    // Create circuit and prove membership...

    Ok(())
}
```

---

## Best Practices

### 1. Minimize Constraints

```rust
// ❌ Bad: Many separate constraints
let constraints = vec![
    constraint_1,
    constraint_2,
    // ... 100 similar constraints
];

// ✅ Good: Batched/vectorized constraints
let constraint = batched_constraint(inputs, 100);
```

### 2. Use ZK-Friendly Primitives

```rust
// ❌ Expensive: SHA256 (~20,000 constraints)
let hash = SHA256Gadget::hash(preimage);

// ✅ Efficient: Poseidon (~300 constraints)
let hash = PoseidonGadget::hash(preimage);
```

### 3. Leverage Lookup Tables

```rust
// ❌ Expensive: Compute in-circuit
let result = expensive_computation(input);

// ✅ Efficient: Precomputed lookup
let result = lookup_table.get(input);
```

### 4. Proper Witness Generation

```rust
// ❌ Bad: Witness computed incorrectly
let witness = Witness::custom(wrong_data);

// ✅ Good: Always validate
let witness = Witness::custom(data);
assert!(witness.satisfies_statement(&statement));
```

### 5. Security Considerations

```rust
// ❌ Dangerous: Reusing randomness
let r = vec![0u8; 32]; // Same every time!

// ✅ Safe: Fresh randomness
let mut r = vec![0u8; 32];
rand::thread_rng().fill(&mut r[..]);
```

### 6. Testing Circuits

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_circuit_satisfiability() {
        // Test with valid inputs
        let valid_witness = generate_valid_witness();
        assert!(circuit.is_satisfied(&valid_witness));

        // Test with invalid inputs
        let invalid_witness = generate_invalid_witness();
        assert!(!circuit.is_satisfied(&invalid_witness));
    }

    #[test]
    fn test_constraint_count() {
        let circuit = build_circuit();
        let constraint_count = circuit.constraints().len();

        // Verify constraint count is reasonable
        assert!(constraint_count < MAX_CONSTRAINTS);
    }
}
```

---

## See Also

- [ZK Proof API Reference](./ZK_PROOF_API.md) - Complete API documentation
- [Integration Guide](./INTEGRATION_GUIDE.md) - Step-by-step integration
- [Bulletproofs Implementation](./BULLETPROOFS_IMPLEMENTATION.md) - Range proof details
- [Groth16 Trusted Setup](./GROTH16_TRUSTED_SETUP.md) - Setup ceremony
