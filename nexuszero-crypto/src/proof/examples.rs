//! Example usage of the modular ZK proof architecture
//!
//! This example demonstrates how to use the new trait-based prover,
//! verifier, and witness generation systems for improved modularity.

use crate::proof::*;
use crate::CryptoResult;

/// Example: Using the modular prover system
pub async fn example_modular_proving() -> CryptoResult<()> {
    // Create a statement
    let statement = Statement::new(StatementType::DiscreteLog {
        generator: vec![2; 32], // g = 2
        public_value: vec![0x12; 32], // Some public value
    });

    // Create a witness
    let witness = Witness::discrete_log(vec![0x34; 32]); // Secret exponent

    // Create prover registry and register provers
    let mut prover_registry = ProverRegistry::new();
    prover_registry.register(Box::new(prover::DirectProver));
    prover_registry.register(Box::new(prover::CircuitProver {
        engine: circuit::CircuitEngine::new(),
    }));

    // Select and use a prover
    let prover = prover_registry.get("direct")
        .ok_or_else(|| crate::CryptoError::InvalidParameter("Prover not found".to_string()))?;

    let config = ProverConfig {
        security_level: crate::SecurityLevel::High,
        optimizations: std::collections::HashMap::new(),
        backend_params: std::collections::HashMap::new(),
    };

    // Generate proof
    let proof = prover.prove(&statement, &witness, &config).await?;

    println!("Generated proof with {} bytes", proof.size());

    // Create verifier registry
    let mut verifier_registry = VerifierRegistry::new();
    verifier_registry.register(Box::new(verifier::DirectVerifier));
    verifier_registry.register(Box::new(verifier::HardwareVerifier {
        device_type: verifier::HardwareType::GPU,
    }));

    // Select optimal verifier
    let requirements = VerificationRequirements {
        max_proof_size: 4096,
        max_latency_ms: 50,
        no_trusted_setup_required: true,
        security_level: crate::SecurityLevel::High,
    };

    let verifier = verifier_registry.select_optimal(&requirements)
        .ok_or_else(|| crate::CryptoError::InvalidParameter("No suitable verifier found".to_string()))?;

    let verifier_config = VerifierConfig {
        security_level: crate::SecurityLevel::High,
        optimizations: std::collections::HashMap::new(),
        backend_params: std::collections::HashMap::new(),
    };

    // Verify proof
    let is_valid = verifier.verify(&statement, &proof, &verifier_config).await?;

    println!("Proof verification: {}", is_valid);

    Ok(())
}

/// Example: Using the witness generation DSL
pub async fn example_witness_dsl() -> CryptoResult<()> {
    // Create a witness generation plan
    let plan = WitnessGenerationPlan {
        statement_type: StatementType::Preimage {
            hash_function: statement::HashFunction::SHA3_256,
            hash_output: vec![0xAB; 32], // Some hash output
        },
        strategy: WitnessStrategy::Direct(DirectStrategy {
            transformation: DataTransformation::Hash {
                algorithm: HashAlgorithm::SHA3_256,
            },
        }),
        validation_rules: vec![ValidationRule::SatisfiesStatement],
    };

    // Build witness generator
    let generator = WitnessBuilder::new()
        .add_plan(plan.statement_type.clone(), plan)
        .build();

    // Generate witness
    let inputs = std::collections::HashMap::from([
        ("secret".to_string(), vec![0x12; 32]), // Input data
    ]);

    let statement = Statement::new(plan.statement_type);
    let witness = generator.generate_witness(&statement, inputs).await?;

    println!("Generated witness of type: {:?}", witness.witness_type());

    Ok(())
}

/// Example: Composing circuits from components
pub async fn example_circuit_composition() -> CryptoResult<()> {
    // Create circuit components
    struct HashComponent;
    struct RangeComponent;

    #[async_trait::async_trait]
    impl CircuitComponent for HashComponent {
        fn id(&self) -> &str { "hash" }
        fn inputs(&self) -> Vec<Variable> {
            vec![Variable {
                name: "input".to_string(),
                var_type: VariableType::Bytes { length: 32 },
                bit_length: 256,
            }]
        }
        fn outputs(&self) -> Vec<Variable> {
            vec![Variable {
                name: "hash".to_string(),
                var_type: VariableType::Bytes { length: 32 },
                bit_length: 256,
            }]
        }
        fn constraints(&self) -> Vec<Constraint> {
            vec![Constraint {
                left: Expression::Variable("hash".to_string()),
                right: Expression::Variable("input".to_string()), // Simplified
                constraint_type: ConstraintType::Equality,
            }]
        }

        async fn generate_witness(&self, inputs: &std::collections::HashMap<String, Vec<u8>>) -> CryptoResult<WitnessData> {
            let input = inputs.get("input").unwrap();
            use sha3::{Digest, Sha3_256};
            let mut hasher = Sha3_256::new();
            hasher.update(input);
            let hash = hasher.finalize().to_vec();

            Ok(WitnessData {
                variables: std::collections::HashMap::from([
                    ("hash".to_string(), hash),
                ]),
                randomness: vec![0; 32],
            })
        }
    }

    // Create circuit engine and register components
    let mut engine = CircuitEngine::new();
    engine.register_component(Box::new(HashComponent));

    // Define a circuit
    let circuit = Circuit {
        id: "hash_circuit".to_string(),
        components: vec!["hash".to_string()],
        connections: vec![], // No connections needed for single component
        public_inputs: vec!["input".to_string()],
        private_inputs: vec![],
        outputs: vec!["hash".to_string()],
    };

    engine.register_circuit(circuit);

    // Use circuit in proving (would be integrated with prover system)
    println!("Circuit composition example completed");

    Ok(())
}

/// Example: Custom proof system implementation
pub struct CustomProofSystem;

#[async_trait::async_trait]
impl Prover for CustomProofSystem {
    fn id(&self) -> &str { "custom" }

    fn supported_statements(&self) -> Vec<StatementType> {
        vec![StatementType::Custom {
            description: "Custom proof type".to_string(),
        }]
    }

    async fn prove(&self, _statement: &Statement, _witness: &Witness, _config: &ProverConfig) -> CryptoResult<Proof> {
        // Implement custom proving logic
        Err(crate::CryptoError::NotImplemented("Custom proving not implemented".to_string()))
    }

    async fn prove_batch(&self, _statements: &[Statement], _witnesses: &[Witness], _config: &ProverConfig) -> CryptoResult<Vec<Proof>> {
        Err(crate::CryptoError::NotImplemented("Custom batch proving not implemented".to_string()))
    }

    fn capabilities(&self) -> ProverCapabilities {
        ProverCapabilities {
            max_proof_size: 1024,
            avg_proving_time_ms: 100,
            trusted_setup_required: false,
            zk_guarantee: ZKGuarantee::Computational,
            supported_optimizations: vec!["custom".to_string()],
        }
    }
}

/// Run all examples
pub async fn run_all_examples() -> CryptoResult<()> {
    println!("Running modular architecture examples...");

    example_modular_proving().await?;
    example_witness_dsl().await?;
    example_circuit_composition().await?;

    println!("All examples completed successfully!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_modular_examples() {
        run_all_examples().await.unwrap();
    }
}</content>
<parameter name="filePath">c:\Users\sgbil\Nexuszero-Protocol\nexuszero-crypto\src\proof\examples.rs