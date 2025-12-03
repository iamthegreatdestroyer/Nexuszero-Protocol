# ZK Proof System - Practical Examples

> Comprehensive code examples for the NexusZero ZK proof system

## Table of Contents

1. [Quick Start Examples](#quick-start-examples)
2. [Authentication Proofs](#authentication-proofs)
3. [Range Proofs](#range-proofs)
4. [Circuit-Based Proofs](#circuit-based-proofs)
5. [Cross-Chain Verification](#cross-chain-verification)
6. [Privacy-Preserving Transactions](#privacy-preserving-transactions)
7. [Advanced Patterns](#advanced-patterns)
8. [Error Handling](#error-handling)
9. [Testing Examples](#testing-examples)

---

## Quick Start Examples

### Example 1: Basic Discrete Log Proof

Prove knowledge of a secret `x` such that `y = g^x mod p`:

```rust
use nexuszero_crypto::proof::{
    Statement, StatementType, StatementBuilder,
    Witness, WitnessType, SecretData,
    prove, verify, ProverConfig, VerifierConfig,
};
use num_bigint::BigUint;

fn discrete_log_example() -> Result<(), Box<dyn std::error::Error>> {
    // Public parameters (would come from trusted setup)
    let generator = BigUint::from(2u32);
    let modulus = BigUint::parse_bytes(
        b"FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD1\
          29024E088A67CC74020BBEA63B139B22514A08798E3404DD\
          EF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245\
          E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7ED\
          EE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3D\
          C2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F\
          83655D23DCA3AD961C62F356208552BB9ED529077096966D\
          670C354E4ABC9804F1746C08CA237327FFFFFFFFFFFFFFFF",
        16
    ).unwrap();

    // The secret we want to prove knowledge of
    let secret = BigUint::from(12345678u64);

    // Compute public value: y = g^x mod p
    let public_value = generator.modpow(&secret, &modulus);

    // Build the statement (public claim)
    let statement = StatementBuilder::new("dlog_proof")
        .discrete_log(
            generator.clone(),
            public_value.clone(),
            modulus.clone(),
        )
        .build()?;

    // Create the witness (secret knowledge)
    let witness = Witness::new(
        WitnessType::DiscreteLog {
            exponent: secret,
            generator,
            modulus: modulus.clone(),
        },
        SecretData::BigUint(secret.clone()),
    );

    // Prove knowledge
    let config = ProverConfig::default();
    let proof = prove(&statement, &witness, &config)?;

    // Verify the proof
    let verifier_config = VerifierConfig::default();
    let is_valid = verify(&statement, &proof, &verifier_config)?;

    assert!(is_valid, "Proof should be valid");
    println!("✓ Discrete log proof verified successfully!");

    Ok(())
}
```

### Example 2: Hash Preimage Proof

Prove knowledge of a preimage for a hash:

```rust
use nexuszero_crypto::proof::{
    Statement, StatementBuilder, Witness, WitnessType,
    SecretData, HashFunction, prove, verify,
};
use sha2::{Sha256, Digest};

fn preimage_example() -> Result<(), Box<dyn std::error::Error>> {
    // The secret preimage (e.g., a password)
    let preimage = b"my_secret_password_123";

    // Compute the hash
    let mut hasher = Sha256::new();
    hasher.update(preimage);
    let hash = hasher.finalize().to_vec();

    // Build statement: "I know a value that hashes to this"
    let statement = StatementBuilder::new("password_proof")
        .preimage(hash.clone(), HashFunction::Sha256)
        .build()?;

    // Create witness with the secret preimage
    let witness = Witness::new(
        WitnessType::Preimage {
            preimage: preimage.to_vec(),
            hash_function: HashFunction::Sha256,
        },
        SecretData::Bytes(preimage.to_vec()),
    );

    // Generate and verify proof
    let proof = prove(&statement, &witness, &Default::default())?;
    let valid = verify(&statement, &proof, &Default::default())?;

    assert!(valid);
    println!("✓ Preimage proof verified!");

    Ok(())
}
```

---

## Authentication Proofs

### Example 3: Password-Based Authentication Without Revealing Password

```rust
use nexuszero_crypto::proof::{
    Statement, StatementBuilder, Witness, WitnessType,
    SecretData, HashFunction, prove, verify,
};
use argon2::{Argon2, password_hash::SaltString};

/// Authenticate a user without revealing their password
struct ZkAuthenticator {
    // In production: stored in secure database
    stored_hashes: std::collections::HashMap<String, Vec<u8>>,
}

impl ZkAuthenticator {
    pub fn new() -> Self {
        Self {
            stored_hashes: std::collections::HashMap::new(),
        }
    }

    /// Register a new user (one-time setup)
    pub fn register(&mut self, username: &str, password: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
        // Use Argon2 for password hashing (slow hash for security)
        let salt = SaltString::generate(&mut rand::thread_rng());
        let argon2 = Argon2::default();

        let hash = argon2.hash_password(password, &salt)?;
        self.stored_hashes.insert(
            username.to_string(),
            hash.hash.unwrap().as_bytes().to_vec(),
        );

        println!("User '{}' registered", username);
        Ok(())
    }

    /// Generate a ZK proof of password knowledge
    pub fn create_auth_proof(
        &self,
        username: &str,
        password: &[u8],
    ) -> Result<Proof, Box<dyn std::error::Error>> {
        let stored_hash = self.stored_hashes
            .get(username)
            .ok_or("User not found")?;

        // Statement: "I know the preimage of this hash"
        let statement = StatementBuilder::new(&format!("auth_{}", username))
            .preimage(stored_hash.clone(), HashFunction::Argon2id)
            .build()?;

        // Witness: the actual password
        let witness = Witness::new(
            WitnessType::Preimage {
                preimage: password.to_vec(),
                hash_function: HashFunction::Argon2id,
            },
            SecretData::Bytes(password.to_vec()),
        );

        // Generate proof
        prove(&statement, &witness, &Default::default())
    }

    /// Verify an authentication proof
    pub fn verify_auth(
        &self,
        username: &str,
        proof: &Proof,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        let stored_hash = self.stored_hashes
            .get(username)
            .ok_or("User not found")?;

        let statement = StatementBuilder::new(&format!("auth_{}", username))
            .preimage(stored_hash.clone(), HashFunction::Argon2id)
            .build()?;

        verify(&statement, proof, &Default::default())
    }
}

// Usage
fn authentication_flow() -> Result<(), Box<dyn std::error::Error>> {
    let mut auth = ZkAuthenticator::new();

    // Registration (done once)
    auth.register("alice", b"super_secret_password")?;

    // Authentication (done on each login)
    let proof = auth.create_auth_proof("alice", b"super_secret_password")?;

    // Server verifies without learning the password
    let is_authenticated = auth.verify_auth("alice", &proof)?;

    assert!(is_authenticated);
    println!("✓ Alice authenticated via ZK proof!");

    Ok(())
}
```

### Example 4: Multi-Factor ZK Authentication

```rust
use nexuszero_crypto::proof::{
    Statement, StatementBuilder, Witness, Circuit,
    CircuitComponent, prove, verify,
};

/// MFA combining password + hardware key without revealing either
fn multi_factor_auth_example() -> Result<(), Box<dyn std::error::Error>> {
    // Factor 1: Password hash
    let password = b"user_password";
    let password_hash = sha256_hash(password);

    // Factor 2: Hardware key signature
    let hardware_key_commitment = get_hardware_key_commitment();
    let hardware_signature = sign_with_hardware_key(&challenge);

    // Combined statement: "I know password AND have hardware key"
    let statement = StatementBuilder::new("mfa_auth")
        .preimage(password_hash.clone(), HashFunction::Sha256)
        .add_custom_constraint("hardware_key", |params| {
            params.get("commitment") == Some(&hardware_key_commitment) &&
            params.get("challenge") == Some(&challenge)
        })
        .build()?;

    // Combined witness
    let witness = Witness::composite(vec![
        Witness::preimage(password.to_vec(), HashFunction::Sha256),
        Witness::custom("hardware_key", hardware_signature.into()),
    ]);

    // Single proof covers both factors
    let proof = prove(&statement, &witness, &Default::default())?;

    // Verify
    let valid = verify(&statement, &proof, &Default::default())?;

    assert!(valid);
    println!("✓ Multi-factor authentication succeeded!");

    Ok(())
}
```

---

## Range Proofs

### Example 5: Age Verification (Prove 18+ Without Revealing Age)

```rust
use nexuszero_crypto::proof::{
    Statement, StatementBuilder, Witness, WitnessType,
    SecretData, prove, verify,
};
use num_bigint::BigUint;

/// Prove someone is at least 18 without revealing their actual age
fn age_verification_example() -> Result<(), Box<dyn std::error::Error>> {
    // User's actual age (secret)
    let actual_age = 25u64;

    // Statement: "My age is in the range [18, 150]"
    let statement = StatementBuilder::new("age_verification")
        .range(
            BigUint::from(18u64),   // minimum (inclusive)
            BigUint::from(150u64),  // maximum (inclusive)
        )
        .build()?;

    // Witness: the actual age
    let witness = Witness::new(
        WitnessType::Range {
            value: BigUint::from(actual_age),
            min: BigUint::from(18u64),
            max: BigUint::from(150u64),
        },
        SecretData::BigUint(BigUint::from(actual_age)),
    );

    // Generate proof
    let proof = prove(&statement, &witness, &Default::default())?;

    // Verifier can confirm person is 18+ without learning exact age
    let is_adult = verify(&statement, &proof, &Default::default())?;

    assert!(is_adult);
    println!("✓ Age verification passed (user is 18+)");

    // The verifier ONLY knows: age ∈ [18, 150]
    // The verifier does NOT know: actual age = 25

    Ok(())
}
```

### Example 6: Credit Score Range Proof

```rust
/// Prove credit score is in an acceptable range without revealing exact score
fn credit_score_proof() -> Result<(), Box<dyn std::error::Error>> {
    let actual_score = 742u64; // Secret

    // Different ranges for different loan products
    let ranges = [
        ("standard_loan", 620, 850),
        ("premium_loan", 720, 850),
        ("ultra_premium", 800, 850),
    ];

    for (product, min, max) in ranges {
        let statement = StatementBuilder::new(&format!("{}_eligibility", product))
            .range(BigUint::from(min as u64), BigUint::from(max as u64))
            .build()?;

        let witness = Witness::range(
            BigUint::from(actual_score),
            BigUint::from(min as u64),
            BigUint::from(max as u64),
        );

        // Try to generate proof (will fail if score not in range)
        match prove(&statement, &witness, &Default::default()) {
            Ok(proof) => {
                let valid = verify(&statement, &proof, &Default::default())?;
                if valid {
                    println!("✓ Eligible for {}", product);
                }
            }
            Err(_) => {
                println!("✗ Not eligible for {} (score not in range)", product);
            }
        }
    }

    Ok(())
}
```

### Example 7: Salary Range Proof for Apartment Application

```rust
/// Prove income meets threshold without revealing exact salary
fn income_verification() -> Result<(), Box<dyn std::error::Error>> {
    let monthly_salary = BigUint::from(8500u64); // Secret: $8,500/month
    let rent_amount = BigUint::from(2500u64);    // Public: $2,500/month rent

    // Landlord requires 3x rent as income
    let required_income = &rent_amount * 3u32;   // $7,500 minimum

    // Statement: "My income is >= 3x the rent"
    let statement = StatementBuilder::new("income_verification")
        .range(
            required_income.clone(),
            BigUint::from(1_000_000u64), // Reasonable upper bound
        )
        .build()?;

    let witness = Witness::range(
        monthly_salary,
        required_income,
        BigUint::from(1_000_000u64),
    );

    let proof = prove(&statement, &witness, &Default::default())?;
    let qualifies = verify(&statement, &proof, &Default::default())?;

    if qualifies {
        println!("✓ Income verification passed");
        println!("  Landlord knows: income >= ${}", rent_amount * 3u32);
        println!("  Landlord does NOT know: exact salary");
    }

    Ok(())
}
```

---

## Circuit-Based Proofs

### Example 8: Merkle Tree Membership Proof

```rust
use nexuszero_crypto::proof::{
    Circuit, CircuitBuilder, CircuitComponent, Variable,
    Constraint, ConstraintType, prove, verify,
};

/// Prove membership in a set without revealing which element
fn merkle_membership_proof() -> Result<(), Box<dyn std::error::Error>> {
    // The Merkle root (public)
    let merkle_root = compute_merkle_root(&all_members);

    // The leaf (secret) and path (secret)
    let my_leaf = hash_member(&my_data);
    let merkle_path = get_merkle_path(&my_data);

    // Build circuit for Merkle verification
    let mut circuit = CircuitBuilder::new("merkle_membership");

    // Variables
    let leaf = circuit.add_private_variable("leaf");
    let root = circuit.add_public_variable("root");
    let path_elements: Vec<_> = (0..merkle_path.len())
        .map(|i| circuit.add_private_variable(&format!("path_{}", i)))
        .collect();
    let path_directions: Vec<_> = (0..merkle_path.len())
        .map(|i| circuit.add_private_variable(&format!("dir_{}", i)))
        .collect();

    // Add Merkle verification constraints
    let mut current = leaf;
    for i in 0..merkle_path.len() {
        // Hash(left || right) where order depends on direction
        let left = circuit.add_conditional(
            path_directions[i].clone(),
            current.clone(),
            path_elements[i].clone(),
        );
        let right = circuit.add_conditional(
            path_directions[i].clone(),
            path_elements[i].clone(),
            current.clone(),
        );

        current = circuit.add_hash_constraint(left, right);
    }

    // Final constraint: computed root must equal public root
    circuit.add_constraint(Constraint::new(
        ConstraintType::Equal,
        current,
        root,
    ));

    let circuit = circuit.build()?;

    // Create statement and witness
    let statement = circuit.to_statement()?;
    let witness = circuit.to_witness(|var| {
        match var.name() {
            "leaf" => Some(my_leaf.clone()),
            "root" => Some(merkle_root.clone()),
            name if name.starts_with("path_") => {
                let i: usize = name[5..].parse().ok()?;
                Some(merkle_path[i].sibling.clone())
            }
            name if name.starts_with("dir_") => {
                let i: usize = name[4..].parse().ok()?;
                Some(if merkle_path[i].is_left { 0 } else { 1 })
            }
            _ => None,
        }
    })?;

    // Generate and verify
    let proof = prove(&statement, &witness, &Default::default())?;
    let valid = verify(&statement, &proof, &Default::default())?;

    assert!(valid);
    println!("✓ Merkle membership verified!");

    Ok(())
}
```

### Example 9: Sudoku Solution Proof

```rust
/// Prove you know a valid Sudoku solution without revealing it
fn sudoku_proof_example() -> Result<(), Box<dyn std::error::Error>> {
    let puzzle = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9],
    ];

    let solution = [
        [5, 3, 4, 6, 7, 8, 9, 1, 2],
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [1, 9, 8, 3, 4, 2, 5, 6, 7],
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [4, 2, 6, 8, 5, 3, 7, 9, 1],
        [7, 1, 3, 9, 2, 4, 8, 5, 6],
        [9, 6, 1, 5, 3, 7, 2, 8, 4],
        [2, 8, 7, 4, 1, 9, 6, 3, 5],
        [3, 4, 5, 2, 8, 6, 1, 7, 9],
    ];

    // Build circuit
    let mut circuit = CircuitBuilder::new("sudoku_solution");

    // Add variables for each cell
    let cells: [[Variable; 9]; 9] = array_init::array_init(|i| {
        array_init::array_init(|j| {
            if puzzle[i][j] != 0 {
                circuit.add_public_variable(&format!("cell_{}_{}", i, j))
            } else {
                circuit.add_private_variable(&format!("cell_{}_{}", i, j))
            }
        })
    });

    // Constraint 1: Each cell is 1-9
    for i in 0..9 {
        for j in 0..9 {
            circuit.add_constraint(Constraint::new(
                ConstraintType::Range,
                cells[i][j].clone(),
                Variable::constant(1, 9),
            ));
        }
    }

    // Constraint 2: Each row has unique values
    for i in 0..9 {
        for j in 0..9 {
            for k in (j + 1)..9 {
                circuit.add_constraint(Constraint::new(
                    ConstraintType::NotEqual,
                    cells[i][j].clone(),
                    cells[i][k].clone(),
                ));
            }
        }
    }

    // Constraint 3: Each column has unique values
    for j in 0..9 {
        for i in 0..9 {
            for k in (i + 1)..9 {
                circuit.add_constraint(Constraint::new(
                    ConstraintType::NotEqual,
                    cells[i][j].clone(),
                    cells[k][j].clone(),
                ));
            }
        }
    }

    // Constraint 4: Each 3x3 box has unique values
    for box_row in 0..3 {
        for box_col in 0..3 {
            let mut box_cells = Vec::new();
            for i in 0..3 {
                for j in 0..3 {
                    box_cells.push(cells[box_row * 3 + i][box_col * 3 + j].clone());
                }
            }
            for i in 0..9 {
                for j in (i + 1)..9 {
                    circuit.add_constraint(Constraint::new(
                        ConstraintType::NotEqual,
                        box_cells[i].clone(),
                        box_cells[j].clone(),
                    ));
                }
            }
        }
    }

    let circuit = circuit.build()?;

    // Generate proof
    let statement = circuit.to_statement_with_values(|var| {
        let (i, j) = parse_cell_name(var.name())?;
        if puzzle[i][j] != 0 {
            Some(puzzle[i][j] as u64)
        } else {
            None
        }
    })?;

    let witness = circuit.to_witness(|var| {
        let (i, j) = parse_cell_name(var.name())?;
        Some(solution[i][j] as u64)
    })?;

    let proof = prove(&statement, &witness, &Default::default())?;
    let valid = verify(&statement, &proof, &Default::default())?;

    assert!(valid);
    println!("✓ Sudoku solution verified without revealing it!");

    Ok(())
}
```

---

## Cross-Chain Verification

### Example 10: Cross-Chain Asset Transfer Proof

```rust
use nexuszero_crypto::proof::{
    Statement, StatementBuilder, Witness, prove, verify,
};
use nexuszero_integration::chain_connectors::{
    EthereumConnector, SolanaConnector, ChainState,
};

/// Prove ownership on one chain to claim on another
async fn cross_chain_transfer() -> Result<(), Box<dyn std::error::Error>> {
    let eth_connector = EthereumConnector::new(&config.ethereum)?;
    let sol_connector = SolanaConnector::new(&config.solana)?;

    // Step 1: Lock assets on Ethereum
    let lock_tx = eth_connector.lock_assets(
        amount: 1_000_000,
        asset: "USDC",
        recipient_chain: "solana",
    ).await?;

    // Step 2: Get proof of lock on Ethereum
    let eth_proof_data = eth_connector.get_inclusion_proof(&lock_tx.hash).await?;

    // Step 3: Build ZK proof of Ethereum state
    let statement = StatementBuilder::new("cross_chain_transfer")
        .add_custom_constraint("ethereum_lock", |params| {
            // Verify Merkle inclusion in Ethereum state root
            params.contains_key("state_root") &&
            params.contains_key("inclusion_proof") &&
            params.contains_key("lock_data")
        })
        .build()?;

    let witness = Witness::custom("ethereum_lock", json!({
        "tx_hash": lock_tx.hash,
        "amount": 1_000_000,
        "asset": "USDC",
        "merkle_proof": eth_proof_data.proof,
        "block_header": eth_proof_data.block_header,
    }))?;

    // Step 4: Generate proof
    let proof = prove(&statement, &witness, &Default::default())?;

    // Step 5: Submit proof to Solana for minting
    let mint_tx = sol_connector.mint_with_proof(
        proof: &proof,
        statement: &statement,
        recipient: &my_solana_address,
    ).await?;

    println!("✓ Cross-chain transfer complete!");
    println!("  Ethereum lock: {}", lock_tx.hash);
    println!("  Solana mint: {}", mint_tx.signature);

    Ok(())
}
```

### Example 11: Multi-Chain Balance Aggregation Proof

```rust
/// Prove total balance across chains without revealing individual balances
async fn aggregate_balance_proof() -> Result<(), Box<dyn std::error::Error>> {
    // Get balances from multiple chains (secrets)
    let eth_balance = eth_connector.get_balance(&my_address).await?;
    let sol_balance = sol_connector.get_balance(&my_pubkey).await?;
    let btc_balance = btc_connector.get_balance(&my_btc_address).await?;

    // Total balance
    let total = eth_balance + sol_balance + btc_balance;

    // Minimum required (public)
    let minimum_required = BigUint::from(100_000u64); // $100k equivalent

    // Build statement: "My total cross-chain balance >= $100k"
    let statement = StatementBuilder::new("cross_chain_balance")
        .range(
            minimum_required.clone(),
            BigUint::from(1_000_000_000u64), // $1B upper bound
        )
        .add_custom_constraint("balance_sum", |params| {
            // Constraint: eth + sol + btc = total
            true // Simplified; real impl verifies sum
        })
        .build()?;

    // Witness: individual balances and total
    let witness = Witness::composite(vec![
        Witness::custom("eth_balance", eth_balance.into()),
        Witness::custom("sol_balance", sol_balance.into()),
        Witness::custom("btc_balance", btc_balance.into()),
        Witness::range(total, minimum_required, BigUint::from(1_000_000_000u64)),
    ]);

    let proof = prove(&statement, &witness, &Default::default())?;

    // Verifier learns: total >= $100k
    // Verifier does NOT learn: individual chain balances

    println!("✓ Cross-chain balance proof generated");

    Ok(())
}
```

---

## Privacy-Preserving Transactions

### Example 12: Private Token Transfer

```rust
/// Transfer tokens without revealing amount or recipient
fn private_transfer() -> Result<(), Box<dyn std::error::Error>> {
    // Sender's note (private)
    let sender_note = Note {
        owner: my_public_key,
        amount: BigUint::from(1000u64),
        blinding: random_scalar(),
    };

    // Transfer amount (private)
    let transfer_amount = BigUint::from(300u64);

    // Recipient (public commitment only)
    let recipient_commitment = pedersen_commit(
        recipient_public_key,
        random_scalar(),
    );

    // Build circuit for private transfer
    let mut circuit = CircuitBuilder::new("private_transfer");

    // Input note
    let input_owner = circuit.add_private_variable("input_owner");
    let input_amount = circuit.add_private_variable("input_amount");
    let input_blinding = circuit.add_private_variable("input_blinding");
    let input_commitment = circuit.add_public_variable("input_commitment");

    // Output notes
    let output1_owner = circuit.add_private_variable("output1_owner");
    let output1_amount = circuit.add_private_variable("output1_amount");
    let output1_blinding = circuit.add_private_variable("output1_blinding");
    let output1_commitment = circuit.add_public_variable("output1_commitment");

    let output2_owner = circuit.add_private_variable("output2_owner");
    let output2_amount = circuit.add_private_variable("output2_amount");
    let output2_blinding = circuit.add_private_variable("output2_blinding");
    let output2_commitment = circuit.add_public_variable("output2_commitment");

    // Constraint 1: Input commitment is valid
    circuit.add_constraint(Constraint::pedersen_commitment(
        input_commitment.clone(),
        input_amount.clone(),
        input_blinding.clone(),
    ));

    // Constraint 2: Output commitments are valid
    circuit.add_constraint(Constraint::pedersen_commitment(
        output1_commitment.clone(),
        output1_amount.clone(),
        output1_blinding.clone(),
    ));
    circuit.add_constraint(Constraint::pedersen_commitment(
        output2_commitment.clone(),
        output2_amount.clone(),
        output2_blinding.clone(),
    ));

    // Constraint 3: Conservation of value (input = output1 + output2)
    circuit.add_constraint(Constraint::new(
        ConstraintType::Equal,
        input_amount.clone(),
        Expression::add(output1_amount.clone(), output2_amount.clone()),
    ));

    // Constraint 4: Amounts are non-negative
    circuit.add_constraint(Constraint::range(output1_amount.clone(), 0, u64::MAX));
    circuit.add_constraint(Constraint::range(output2_amount.clone(), 0, u64::MAX));

    // Constraint 5: Sender owns the input
    circuit.add_constraint(Constraint::signature_check(
        input_owner.clone(),
        my_signature,
    ));

    let circuit = circuit.build()?;

    // Generate proof
    let proof = prove_circuit(&circuit, &values)?;

    // Submit transaction (only commitments and proof are public)
    let tx = Transaction {
        input_commitment: sender_note.commitment(),
        output_commitments: vec![
            recipient_note.commitment(),
            change_note.commitment(),
        ],
        proof,
    };

    Ok(())
}
```

### Example 13: Voting Without Revealing Vote

```rust
/// Cast a vote that can be tallied without revealing individual choices
fn private_voting() -> Result<(), Box<dyn std::error::Error>> {
    // Voter's choice (secret): 0 = No, 1 = Yes
    let my_vote = 1u64;

    // Voter's registration proof (from census Merkle tree)
    let voter_id = hash(my_public_key);
    let census_proof = get_census_merkle_proof(&voter_id);

    // Build statement
    let statement = StatementBuilder::new("private_vote")
        // Prove: I'm in the census
        .add_custom_constraint("census_membership", |_| true)
        // Prove: My vote is valid (0 or 1)
        .range(BigUint::from(0u64), BigUint::from(1u64))
        // Public: Encrypted vote for homomorphic tallying
        .add_custom_constraint("encrypted_vote", |_| true)
        .build()?;

    // Encrypt vote for homomorphic addition
    let encrypted_vote = encrypt_for_tally(&my_vote, &election_public_key);

    // Witness
    let witness = Witness::composite(vec![
        Witness::custom("voter_key", my_private_key.into()),
        Witness::custom("census_proof", census_proof.into()),
        Witness::range(
            BigUint::from(my_vote),
            BigUint::from(0u64),
            BigUint::from(1u64),
        ),
    ]);

    let proof = prove(&statement, &witness, &Default::default())?;

    // Submit vote
    let ballot = Ballot {
        nullifier: hash(&my_private_key, &election_id), // Prevents double voting
        encrypted_vote,
        proof,
    };

    submit_ballot(&ballot)?;

    println!("✓ Vote cast anonymously!");
    // Election authority can tally encrypted votes
    // Individual votes remain private

    Ok(())
}
```

---

## Advanced Patterns

### Example 14: Recursive Proof Aggregation

```rust
/// Aggregate multiple proofs into one for efficient verification
fn recursive_proof_aggregation() -> Result<(), Box<dyn std::error::Error>> {
    // Collection of proofs to aggregate
    let proofs: Vec<Proof> = vec![
        proof_a, // From user A
        proof_b, // From user B
        proof_c, // From user C
        // ... potentially thousands
    ];

    // Aggregate recursively
    let mut current_proof = proofs[0].clone();

    for next_proof in proofs.into_iter().skip(1) {
        // Create circuit that verifies both proofs
        let mut circuit = CircuitBuilder::new("recursive_verifier");

        // Verify current aggregated proof
        circuit.add_verification_gadget(
            "verify_current",
            current_proof.statement().clone(),
            current_proof.clone(),
        );

        // Verify next proof
        circuit.add_verification_gadget(
            "verify_next",
            next_proof.statement().clone(),
            next_proof.clone(),
        );

        let circuit = circuit.build()?;

        // Generate proof of valid verification
        let statement = circuit.to_statement()?;
        let witness = circuit.verification_witness()?;

        current_proof = prove(&statement, &witness, &Default::default())?;
    }

    // Final proof verifies ALL original proofs
    let aggregated_proof = current_proof;

    println!("✓ Aggregated {} proofs into 1", proofs.len());
    println!("  Original verification cost: {} ops", proofs.len() * 100_000);
    println!("  Aggregated verification cost: {} ops", 100_000);

    Ok(())
}
```

### Example 15: Proof with Time-Lock

```rust
/// Proof that becomes invalid after a certain time
fn time_locked_proof() -> Result<(), Box<dyn std::error::Error>> {
    let current_timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)?
        .as_secs();

    let expiry_timestamp = current_timestamp + 3600; // Valid for 1 hour

    // Statement includes time constraint
    let statement = StatementBuilder::new("time_locked_auth")
        .preimage(password_hash.clone(), HashFunction::Sha256)
        .add_custom_constraint("time_bound", |params| {
            params.get("max_timestamp") == Some(&expiry_timestamp)
        })
        .build()?;

    let witness = Witness::composite(vec![
        Witness::preimage(password.to_vec(), HashFunction::Sha256),
        Witness::custom("timestamp", current_timestamp.into()),
    ]);

    let proof = prove(&statement, &witness, &Default::default())?;

    // Verification includes time check
    let config = VerifierConfig::default()
        .with_time_check(|statement_time, verify_time| {
            verify_time <= statement_time.max_timestamp
        });

    let valid = verify(&statement, &proof, &config)?;

    // After 1 hour, same proof will fail verification

    Ok(())
}
```

---

## Error Handling

### Example 16: Comprehensive Error Handling

```rust
use nexuszero_crypto::proof::{
    ProofError, ProverError, VerifierError,
    prove, verify, Statement, Witness,
};

fn robust_proof_generation(
    statement: &Statement,
    witness: &Witness,
) -> Result<Proof, Box<dyn std::error::Error>> {
    // Validate inputs first
    if !statement.is_valid() {
        return Err(ProofError::InvalidStatement {
            reason: "Statement validation failed".into(),
            details: statement.validation_errors(),
        }.into());
    }

    if !witness.satisfies_statement(statement)? {
        return Err(ProofError::WitnessStatementMismatch {
            statement_type: statement.statement_type().clone(),
            witness_type: witness.witness_type().clone(),
        }.into());
    }

    // Try proof generation with retries for transient errors
    let mut last_error = None;
    for attempt in 1..=3 {
        match prove(statement, witness, &Default::default()) {
            Ok(proof) => {
                // Sanity check: verify our own proof
                if verify(statement, &proof, &Default::default())? {
                    return Ok(proof);
                } else {
                    return Err(ProofError::SelfVerificationFailed.into());
                }
            }
            Err(e) => {
                match &e {
                    ProverError::InsufficientRandomness => {
                        // Retry with fresh randomness
                        eprintln!("Attempt {}: Insufficient randomness, retrying...", attempt);
                        last_error = Some(e);
                        continue;
                    }
                    ProverError::MemoryExhausted { required, available } => {
                        // Cannot recover, fail immediately
                        return Err(ProofError::ResourceExhausted {
                            resource: "memory",
                            required: *required,
                            available: *available,
                        }.into());
                    }
                    ProverError::TimeoutExceeded { elapsed, limit } => {
                        eprintln!(
                            "Attempt {}: Timeout ({:?} > {:?}), retrying...",
                            attempt, elapsed, limit
                        );
                        last_error = Some(e);
                        continue;
                    }
                    _ => {
                        return Err(e.into());
                    }
                }
            }
        }
    }

    Err(last_error.unwrap().into())
}

fn robust_verification(
    statement: &Statement,
    proof: &Proof,
) -> Result<bool, Box<dyn std::error::Error>> {
    // Set verification timeout
    let config = VerifierConfig::default()
        .with_timeout(Duration::from_secs(30))
        .with_strict_mode(true);

    match verify(statement, proof, &config) {
        Ok(true) => {
            println!("✓ Proof verified successfully");
            Ok(true)
        }
        Ok(false) => {
            println!("✗ Proof verification failed");
            Ok(false)
        }
        Err(VerifierError::InvalidProofFormat { reason }) => {
            eprintln!("Invalid proof format: {}", reason);
            Err(VerifierError::InvalidProofFormat { reason }.into())
        }
        Err(VerifierError::StatementMismatch) => {
            eprintln!("Proof was generated for a different statement");
            Ok(false)
        }
        Err(VerifierError::TimeoutExceeded { .. }) => {
            eprintln!("Verification timed out - proof may be malicious");
            Ok(false)
        }
        Err(e) => Err(e.into()),
    }
}
```

---

## Testing Examples

### Example 17: Unit Testing Proofs

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use nexuszero_crypto::proof::testing::{
        random_statement, random_witness, ProofTestHarness,
    };

    #[test]
    fn test_discrete_log_proof_roundtrip() {
        let harness = ProofTestHarness::new();

        // Generate random valid statement/witness pair
        let (statement, witness) = harness.discrete_log_pair();

        // Prove
        let proof = prove(&statement, &witness, &Default::default())
            .expect("Proof generation should succeed");

        // Verify
        let valid = verify(&statement, &proof, &Default::default())
            .expect("Verification should complete");

        assert!(valid, "Valid proof should verify");
    }

    #[test]
    fn test_invalid_witness_fails() {
        let harness = ProofTestHarness::new();

        let statement = harness.random_preimage_statement();
        let wrong_witness = harness.random_preimage_witness(); // Wrong preimage

        // Proof generation should fail or produce invalid proof
        match prove(&statement, &wrong_witness, &Default::default()) {
            Err(_) => (), // Expected: proof generation fails
            Ok(proof) => {
                // If proof was generated, it should not verify
                let valid = verify(&statement, &proof, &Default::default())
                    .unwrap_or(false);
                assert!(!valid, "Wrong witness should not produce valid proof");
            }
        }
    }

    #[test]
    fn test_tampered_proof_fails() {
        let harness = ProofTestHarness::new();
        let (statement, witness) = harness.range_pair();

        let mut proof = prove(&statement, &witness, &Default::default())
            .expect("Proof generation should succeed");

        // Tamper with proof data
        proof.data[0] ^= 0xFF;

        // Tampered proof should fail verification
        let valid = verify(&statement, &proof, &Default::default())
            .unwrap_or(false);

        assert!(!valid, "Tampered proof should not verify");
    }

    #[test]
    fn test_proof_for_wrong_statement_fails() {
        let harness = ProofTestHarness::new();

        let (statement1, witness1) = harness.discrete_log_pair();
        let (statement2, _) = harness.discrete_log_pair();

        // Generate proof for statement1
        let proof = prove(&statement1, &witness1, &Default::default())
            .expect("Proof generation should succeed");

        // Verify against different statement
        let valid = verify(&statement2, &proof, &Default::default())
            .unwrap_or(false);

        assert!(!valid, "Proof should not verify against different statement");
    }
}
```

### Example 18: Property-Based Testing

```rust
use proptest::prelude::*;

proptest! {
    /// Property: Valid witness always produces valid proof
    #[test]
    fn prop_valid_witness_produces_valid_proof(
        secret in 1u64..1_000_000
    ) {
        let (statement, witness) = create_discrete_log_pair(secret);

        let proof = prove(&statement, &witness, &Default::default())
            .expect("Valid witness should produce proof");

        let valid = verify(&statement, &proof, &Default::default())
            .expect("Verification should complete");

        prop_assert!(valid, "Valid witness should produce valid proof");
    }

    /// Property: Invalid witness never produces valid proof
    #[test]
    fn prop_invalid_witness_never_valid(
        correct_secret in 1u64..1_000_000,
        wrong_secret in 1u64..1_000_000
    ) {
        prop_assume!(correct_secret != wrong_secret);

        let (statement, _) = create_discrete_log_pair(correct_secret);
        let (_, wrong_witness) = create_discrete_log_pair(wrong_secret);

        // Try to prove with wrong witness
        if let Ok(proof) = prove(&statement, &wrong_witness, &Default::default()) {
            let valid = verify(&statement, &proof, &Default::default())
                .unwrap_or(false);
            prop_assert!(!valid, "Wrong witness should not produce valid proof");
        }
    }

    /// Property: Proof is deterministic given same randomness
    #[test]
    fn prop_deterministic_with_seed(
        secret in 1u64..1_000_000,
        seed in any::<u64>()
    ) {
        let (statement, witness) = create_discrete_log_pair(secret);

        let config = ProverConfig::default().with_seed(seed);

        let proof1 = prove(&statement, &witness, &config)
            .expect("Proof generation should succeed");
        let proof2 = prove(&statement, &witness, &config)
            .expect("Proof generation should succeed");

        prop_assert_eq!(proof1, proof2, "Same seed should produce same proof");
    }

    /// Property: Range proof only succeeds when value in range
    #[test]
    fn prop_range_proof_correctness(
        value in 0u64..1_000_000,
        min in 0u64..500_000,
        max in 500_000u64..1_000_000
    ) {
        prop_assume!(min < max);

        let in_range = value >= min && value <= max;

        let statement = create_range_statement(min, max);
        let witness = create_range_witness(value, min, max);

        let result = prove(&statement, &witness, &Default::default());

        if in_range {
            let proof = result.expect("In-range value should produce proof");
            let valid = verify(&statement, &proof, &Default::default())
                .expect("Verification should complete");
            prop_assert!(valid, "In-range proof should verify");
        } else {
            // Out of range should either fail to prove or fail to verify
            if let Ok(proof) = result {
                let valid = verify(&statement, &proof, &Default::default())
                    .unwrap_or(false);
                prop_assert!(!valid, "Out-of-range should not verify");
            }
        }
    }
}
```

### Example 19: Benchmark Tests

```rust
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

fn benchmark_proof_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("proof_generation");

    // Benchmark different statement types
    for statement_type in &["discrete_log", "preimage", "range"] {
        group.bench_with_input(
            BenchmarkId::new("prove", statement_type),
            statement_type,
            |b, &st| {
                let (statement, witness) = match st {
                    "discrete_log" => create_discrete_log_test_case(),
                    "preimage" => create_preimage_test_case(),
                    "range" => create_range_test_case(),
                    _ => unreachable!(),
                };

                b.iter(|| {
                    prove(&statement, &witness, &Default::default())
                        .expect("Proof should succeed")
                });
            },
        );
    }

    group.finish();
}

fn benchmark_verification(c: &mut Criterion) {
    let mut group = c.benchmark_group("verification");

    // Pre-generate proofs
    let test_cases: Vec<_> = ["discrete_log", "preimage", "range"]
        .iter()
        .map(|&st| {
            let (statement, witness) = match st {
                "discrete_log" => create_discrete_log_test_case(),
                "preimage" => create_preimage_test_case(),
                "range" => create_range_test_case(),
                _ => unreachable!(),
            };
            let proof = prove(&statement, &witness, &Default::default()).unwrap();
            (st, statement, proof)
        })
        .collect();

    for (name, statement, proof) in &test_cases {
        group.bench_with_input(
            BenchmarkId::new("verify", name),
            &(statement, proof),
            |b, (st, pr)| {
                b.iter(|| {
                    verify(st, pr, &Default::default())
                        .expect("Verification should complete")
                });
            },
        );
    }

    group.finish();
}

fn benchmark_circuit_compilation(c: &mut Criterion) {
    let mut group = c.benchmark_group("circuit_compilation");

    for constraint_count in [100, 1000, 10000].iter() {
        group.bench_with_input(
            BenchmarkId::new("compile", constraint_count),
            constraint_count,
            |b, &count| {
                let circuit = create_circuit_with_constraints(count);

                b.iter(|| {
                    circuit.compile().expect("Compilation should succeed")
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_proof_generation,
    benchmark_verification,
    benchmark_circuit_compilation,
);
criterion_main!(benches);
```

---

## Quick Reference

### Common Imports

```rust
// Core types
use nexuszero_crypto::proof::{
    Statement, StatementBuilder, StatementType,
    Witness, WitnessType, SecretData,
    Proof, prove, verify,
    ProverConfig, VerifierConfig,
    HashFunction,
};

// Circuit DSL
use nexuszero_crypto::proof::{
    Circuit, CircuitBuilder, CircuitComponent,
    Variable, Constraint, ConstraintType, Expression,
    CircuitEngine,
};

// Witness DSL
use nexuszero_crypto::proof::{
    WitnessBuilder, WitnessStrategy, DataTransformation,
    WitnessGeneratorRegistry,
};

// Errors
use nexuszero_crypto::proof::{
    ProofError, ProverError, VerifierError, CircuitError,
};
```

### Cheat Sheet

| Task                 | Pattern                                                              |
| -------------------- | -------------------------------------------------------------------- |
| Prove knowledge of x | `StatementBuilder::new().discrete_log(g, y, p)`                      |
| Prove hash preimage  | `StatementBuilder::new().preimage(hash, HashFunction::Sha256)`       |
| Prove value in range | `StatementBuilder::new().range(min, max)`                            |
| Build custom circuit | `CircuitBuilder::new().add_variable().add_constraint().build()`      |
| Generate witness     | `WitnessBuilder::new().add_strategy().with_transformation().build()` |
| Configure prover     | `ProverConfig::default().with_security_level(128)`                   |
| Configure verifier   | `VerifierConfig::default().with_timeout(Duration::from_secs(30))`    |

---

## See Also

- [ZK_PROOF_API.md](./ZK_PROOF_API.md) - Complete API reference
- [INTEGRATION_GUIDE.md](./INTEGRATION_GUIDE.md) - Step-by-step integration guide
- [CIRCUIT_DSL.md](./CIRCUIT_DSL.md) - Circuit DSL documentation
- [MONITORING.md](./MONITORING.md) - Metrics and monitoring guide
- [proof-mechanism.md](./proof-mechanism.md) - ZK mechanism design decisions

---

_These examples demonstrate the flexibility and power of the NexusZero ZK proof system. For production use, ensure proper security audits and follow cryptographic best practices._
