use nexuszero_crypto::proof::{
    witness_manager::{WitnessManager, DefaultWitnessManager, WitnessGenerationConfig, ValidationConstraints, RandomnessConfig},
    statement::{StatementBuilder},
};
use std::collections::HashMap;

#[tokio::main]
async fn main() {
    println!("Testing witness manager...");

    let manager = DefaultWitnessManager::new(1000);
    let statement = StatementBuilder::new().discrete_log(vec![1, 2, 3], vec![4, 5, 6]).build().unwrap();
    let secret_data = b"test_secret_data";
    let config = WitnessGenerationConfig {
        parallel_generation: false,
        max_parallel_tasks: 4,
        zero_copy: true,
        randomness_config: RandomnessConfig {
            length: 32,
            secure_random: true,
            custom_entropy: None,
        },
        validation_constraints: ValidationConstraints {
            max_range_value: Some(1000),
            min_range_value: Some(0),
            max_preimage_length: Some(1024),
            required_hash_function: None,
            custom_constraints: HashMap::new(),
        },
    };

    println!("Creating witness...");
    let result = manager.create_witness(&statement, secret_data, &config).await;
    match result {
        Ok(witness) => println!("Success! Witness created: {:?}", witness.id()),
        Err(e) => println!("Error: {:?}", e),
    }
}