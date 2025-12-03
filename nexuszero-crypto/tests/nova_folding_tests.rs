// Copyright (c) 2025 NexusZero Protocol
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of NexusZero Protocol - Advanced Zero-Knowledge Infrastructure
// Licensed under the GNU Affero General Public License v3.0 or later.
// Commercial licensing available at https://nexuszero.io/licensing
//
// NexusZero Protocol™, Privacy Morphing™, and Holographic Proof Compression™
// are trademarks of NexusZero Protocol. All Rights Reserved.

//! Nova Folding Scheme Integration Tests
//!
//! This module contains comprehensive tests for the Nova folding scheme integration.
//! Tests cover:
//! - Basic folding operations
//! - IVC proof generation and verification
//! - Step circuit implementations
//! - R1CS conversion
//! - Proof compression

#![cfg(feature = "nova")]

use nexuszero_crypto::proof::nova::{
    // Core types
    NovaConfig, NovaProver, NovaProof, IVCProof, NovaPublicParams,
    // Circuits
    StepCircuit, TrivialCircuit, MinRootCircuit, HashChainCircuit, CircuitMetadata,
    // R1CS
    R1CSConverter, R1CSConstraintSystem, R1CSInstance, R1CSWitness, R1CSVariable,
    // Folding
    FoldingEngine, FoldedInstance, FoldingProof, FoldingConfig,
    // Types
    NovaError, NovaResult, NovaSecurityLevel, CurveType, NovaMetrics,
};

mod step_circuit_tests {
    use super::*;

    #[test]
    fn test_trivial_circuit_arity() {
        let circuit = TrivialCircuit::new(3);
        assert_eq!(circuit.arity(), 3);
    }

    #[test]
    fn test_trivial_circuit_synthesize() {
        let circuit = TrivialCircuit::new(2);
        let mut cs = R1CSConstraintSystem::new(NovaSecurityLevel::Bit128);
        
        // Allocate inputs
        let z_in: Vec<usize> = (0..2)
            .map(|i| cs.alloc_public(&format!("z_in_{}", i)))
            .collect();
        
        // Synthesize
        let result = circuit.synthesize(&mut cs, 0, &z_in);
        
        assert!(result.is_ok());
        let z_out = result.unwrap();
        assert_eq!(z_out.len(), 2);
        
        // Verify constraints were added
        assert!(cs.num_constraints() >= 2); // At least one equality per output
    }

    #[test]
    fn test_trivial_circuit_compute() {
        let circuit = TrivialCircuit::new(2);
        let input = vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8]];
        
        let output = circuit.compute(0, &input);
        
        assert!(output.is_ok());
        assert_eq!(output.unwrap(), input); // Trivial: output == input
    }

    #[test]
    fn test_trivial_circuit_wrong_arity() {
        let circuit = TrivialCircuit::new(3);
        let mut cs = R1CSConstraintSystem::new(NovaSecurityLevel::Bit128);
        
        // Only 2 inputs when 3 expected
        let z_in: Vec<usize> = (0..2)
            .map(|i| cs.alloc_public(&format!("z_in_{}", i)))
            .collect();
        
        let result = circuit.synthesize(&mut cs, 0, &z_in);
        
        assert!(result.is_err());
    }

    #[test]
    fn test_minroot_circuit_creation() {
        let circuit = MinRootCircuit::new(10);
        assert_eq!(circuit.arity(), 2);
    }

    #[test]
    fn test_minroot_circuit_metadata() {
        let circuit = MinRootCircuit::new(5);
        let metadata = circuit.metadata();
        
        assert!(metadata.name.contains("MinRoot"));
        // 4 constraints per iteration + 1 for y copy
        assert!(metadata.estimated_constraints >= 21);
        assert!(!metadata.parallelizable);
    }

    #[test]
    fn test_hash_chain_circuit() {
        let circuit = HashChainCircuit::poseidon();
        assert_eq!(circuit.arity(), 1);
        
        let metadata = circuit.metadata();
        assert!(metadata.name.contains("Poseidon"));
    }

    #[test]
    fn test_circuit_metadata_defaults() {
        let metadata = CircuitMetadata::default();
        assert!(metadata.name.is_empty());
        assert_eq!(metadata.estimated_constraints, 0);
    }
}

mod r1cs_tests {
    use super::*;

    #[test]
    fn test_r1cs_constraint_system_creation() {
        let cs = R1CSConstraintSystem::new(NovaSecurityLevel::Bit128);
        
        assert_eq!(cs.num_constraints(), 0);
        assert_eq!(cs.num_variables(), 0);
        assert_eq!(cs.num_public_inputs, 0);
        assert_eq!(cs.num_witness, 0);
    }

    #[test]
    fn test_r1cs_variable_allocation() {
        let mut cs = R1CSConstraintSystem::new(NovaSecurityLevel::Bit128);
        
        let pub_var = cs.alloc_public("public_input");
        let priv_var = cs.alloc_private("private_witness");
        
        assert_eq!(cs.num_public_inputs, 1);
        assert_eq!(cs.num_witness, 1);
        assert_eq!(cs.num_variables(), 2);
        assert_ne!(pub_var, priv_var);
    }

    #[test]
    fn test_r1cs_multiplication_constraint() {
        let mut cs = R1CSConstraintSystem::new(NovaSecurityLevel::Bit128);
        
        let a = cs.alloc_public("a");
        let b = cs.alloc_private("b");
        let c = cs.alloc_private("c");
        
        cs.enforce_mul(a, b, c, Some("a_times_b_equals_c"));
        
        assert_eq!(cs.num_constraints(), 1);
    }

    #[test]
    fn test_r1cs_equality_constraint() {
        let mut cs = R1CSConstraintSystem::new(NovaSecurityLevel::Bit128);
        
        let a = cs.alloc_public("a");
        let b = cs.alloc_public("b");
        
        cs.enforce_equal(a, b, Some("a_equals_b"));
        
        assert_eq!(cs.num_constraints(), 1);
    }

    #[test]
    fn test_r1cs_validation_valid() {
        let mut cs = R1CSConstraintSystem::new(NovaSecurityLevel::Bit128);
        
        let a = cs.alloc_public("a");
        let b = cs.alloc_private("b");
        let c = cs.alloc_private("c");
        
        cs.enforce_mul(a, b, c, None);
        
        let result = cs.validate();
        assert!(result.is_ok());
    }

    #[test]
    fn test_r1cs_instance_creation() {
        let instance = R1CSInstance::new(
            vec![vec![1, 2, 3], vec![4, 5, 6]],
            [0u8; 32],
        );
        
        assert_eq!(instance.public_inputs.len(), 2);
    }

    #[test]
    fn test_r1cs_witness_creation() {
        let witness = R1CSWitness::new(vec![vec![1, 2, 3], vec![4, 5, 6]]);
        
        assert_eq!(witness.witness_values.len(), 2);
        assert_eq!(witness.assignments.len(), 2);
    }

    #[test]
    fn test_r1cs_witness_zeroize() {
        let mut witness = R1CSWitness::new(vec![vec![1, 2, 3]]);
        
        witness.zeroize();
        
        for v in &witness.witness_values {
            for byte in v {
                assert_eq!(*byte, 0);
            }
        }
    }
}

mod folding_engine_tests {
    use super::*;

    #[test]
    fn test_folding_engine_creation() {
        let engine = FoldingEngine::new(1000, true);
        
        assert!(engine.accumulator().is_none());
        assert_eq!(engine.metrics().folding_steps, 0);
    }

    #[test]
    fn test_folding_engine_with_config() {
        let config = FoldingConfig {
            security_level: NovaSecurityLevel::Bit256,
            parallel: false,
            batch_size: 32,
            collect_metrics: true,
        };
        
        let engine = FoldingEngine::with_config(config);
        assert!(engine.accumulator().is_none());
    }

    #[test]
    fn test_folding_engine_initialize() {
        let engine = FoldingEngine::new(100, true);
        let mut cs = R1CSConstraintSystem::new(NovaSecurityLevel::Bit128);
        
        cs.alloc_public("x");
        cs.alloc_private("w");
        
        let result = engine.initialize(&cs);
        
        assert!(result.is_ok());
        let folded = result.unwrap();
        assert_eq!(folded.num_steps, 1);
    }

    #[test]
    fn test_folded_instance_initial() {
        let instance = R1CSInstance::new(
            vec![vec![1, 2, 3]],
            [0u8; 32],
        );
        
        let folded = FoldedInstance::initial(&instance);
        
        assert_eq!(folded.num_steps, 1);
        assert_eq!(folded.accumulated_x.len(), 1);
    }

    #[test]
    fn test_folded_instance_relaxed() {
        let instance = R1CSInstance::new(
            vec![vec![1, 2, 3]],
            [0u8; 32],
        );
        
        let folded = FoldedInstance::relaxed(&instance);
        
        // Relaxed instance has extra element for u
        assert_eq!(folded.accumulated_x.len(), 2);
    }

    #[test]
    fn test_folding_proof_size() {
        let proof = FoldingProof {
            cross_term_commitment: vec![0u8; 32],
            challenge: vec![0u8; 32],
            aux_commitments: vec![vec![0u8; 32]],
        };
        
        assert_eq!(proof.size(), 96);
    }

    #[test]
    fn test_folding_verify_basic() {
        let engine = FoldingEngine::new(100, true);
        let mut cs = R1CSConstraintSystem::new(NovaSecurityLevel::Bit128);
        cs.alloc_public("x");
        
        let folded = engine.initialize(&cs).unwrap();
        
        let result = engine.verify(&folded);
        assert!(result.is_ok());
        assert!(result.unwrap());
    }
}

mod nova_prover_tests {
    use super::*;

    #[test]
    fn test_nova_config_default() {
        let config = NovaConfig::default();
        
        assert_eq!(config.max_steps, 1_000_000);
        assert!(config.parallel_witness);
        assert!(config.enable_recursion);
        assert!(matches!(config.security_level, NovaSecurityLevel::Bit128));
    }

    #[test]
    fn test_nova_config_high_security() {
        let config = NovaConfig::high_security();
        
        assert!(matches!(config.security_level, NovaSecurityLevel::Bit256));
    }

    #[test]
    fn test_nova_config_fast_proving() {
        let config = NovaConfig::fast_proving();
        
        assert!(matches!(config.security_level, NovaSecurityLevel::Bit80));
        assert_eq!(config.max_steps, 100_000);
    }

    #[test]
    fn test_nova_prover_creation() {
        let result = NovaProver::new(NovaConfig::default());
        
        assert!(result.is_ok());
    }

    #[test]
    fn test_nova_prover_default() {
        let result = NovaProver::default();
        
        assert!(result.is_ok());
    }

    #[test]
    fn test_nova_prover_setup() {
        let mut prover = NovaProver::new(NovaConfig::default()).unwrap();
        let circuit = TrivialCircuit::new(2);
        
        let result = prover.setup(&circuit);
        
        assert!(result.is_ok());
        let params = result.unwrap();
        assert!(params.primary_shape.num_constraints() > 0);
    }

    #[test]
    fn test_nova_public_params_setup() {
        let circuit = TrivialCircuit::new(2);
        let config = NovaConfig::default();
        
        let result = NovaPublicParams::setup(&circuit, &config);
        
        assert!(result.is_ok());
    }

    #[test]
    fn test_nova_public_params_verify() {
        let circuit = TrivialCircuit::new(2);
        let config = NovaConfig::default();
        
        let params = NovaPublicParams::setup(&circuit, &config).unwrap();
        let is_valid = params.verify_for_circuit(&circuit);
        
        assert!(is_valid.is_ok());
        assert!(is_valid.unwrap());
    }

    #[test]
    fn test_nova_prover_ivc_trivial() {
        let mut prover = NovaProver::new(NovaConfig::default()).unwrap();
        let circuit = TrivialCircuit::new(2);
        
        let initial_state = vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8]];
        let result = prover.prove_ivc(&circuit, &initial_state, 3);
        
        assert!(result.is_ok());
        let proof = result.unwrap();
        assert_eq!(proof.num_steps, 3);
        assert_eq!(proof.initial_state, initial_state);
    }

    #[test]
    fn test_nova_prover_ivc_single_step() {
        let mut prover = NovaProver::new(NovaConfig::default()).unwrap();
        let circuit = TrivialCircuit::new(1);
        
        let initial_state = vec![vec![42]];
        let result = prover.prove_ivc(&circuit, &initial_state, 1);
        
        assert!(result.is_ok());
        let proof = result.unwrap();
        assert_eq!(proof.num_steps, 1);
    }

    #[test]
    fn test_nova_prover_ivc_many_steps() {
        let mut prover = NovaProver::new(NovaConfig::default()).unwrap();
        let circuit = TrivialCircuit::new(1);
        
        let initial_state = vec![vec![1]];
        let result = prover.prove_ivc(&circuit, &initial_state, 100);
        
        assert!(result.is_ok());
        let proof = result.unwrap();
        assert_eq!(proof.num_steps, 100);
    }

    #[test]
    fn test_nova_prover_ivc_wrong_arity() {
        let mut prover = NovaProver::new(NovaConfig::default()).unwrap();
        let circuit = TrivialCircuit::new(3);
        
        // Only 2 state elements when circuit expects 3
        let initial_state = vec![vec![1], vec![2]];
        let result = prover.prove_ivc(&circuit, &initial_state, 1);
        
        assert!(result.is_err());
    }

    #[test]
    fn test_nova_prover_verify_ivc() {
        let mut prover = NovaProver::new(NovaConfig::default()).unwrap();
        let circuit = TrivialCircuit::new(1);
        
        let initial_state = vec![vec![42]];
        let proof = prover.prove_ivc(&circuit, &initial_state, 5).unwrap();
        
        let is_valid = prover.verify_ivc(&circuit, &proof);
        
        assert!(is_valid.is_ok());
        assert!(is_valid.unwrap());
    }

    #[test]
    fn test_nova_prover_compress() {
        let mut prover = NovaProver::new(NovaConfig::default()).unwrap();
        let circuit = TrivialCircuit::new(1);
        
        let initial_state = vec![vec![1]];
        let ivc_proof = prover.prove_ivc(&circuit, &initial_state, 3).unwrap();
        
        let result = prover.compress(&ivc_proof);
        
        assert!(result.is_ok());
        let compressed = result.unwrap();
        assert_eq!(compressed.num_steps, 3);
    }

    #[test]
    fn test_ivc_proof_size() {
        let mut prover = NovaProver::new(NovaConfig::default()).unwrap();
        let circuit = TrivialCircuit::new(1);
        
        let initial_state = vec![vec![1]];
        let proof = prover.prove_ivc(&circuit, &initial_state, 10).unwrap();
        
        assert!(proof.size() > 0);
    }

    #[test]
    fn test_ivc_proof_is_compressed() {
        let mut prover = NovaProver::new(NovaConfig::default()).unwrap();
        let circuit = TrivialCircuit::new(1);
        
        let initial_state = vec![vec![1]];
        let proof = prover.prove_ivc(&circuit, &initial_state, 3).unwrap();
        
        // IVC proof has folding proofs
        assert!(!proof.is_compressed());
    }
}

mod types_tests {
    use super::*;

    #[test]
    fn test_nova_security_level_rounds() {
        assert_eq!(NovaSecurityLevel::Bit128.rounds(), 24);
        assert_eq!(NovaSecurityLevel::Bit192.rounds(), 36);
        assert_eq!(NovaSecurityLevel::Bit256.rounds(), 48);
    }

    #[test]
    fn test_nova_security_level_constraint_multiplier() {
        assert_eq!(NovaSecurityLevel::Bit128.constraint_multiplier(), 1);
        assert_eq!(NovaSecurityLevel::Bit192.constraint_multiplier(), 2);
        assert_eq!(NovaSecurityLevel::Bit256.constraint_multiplier(), 3);
    }

    #[test]
    fn test_curve_type_scalar_bits() {
        assert_eq!(CurveType::Pallas.scalar_bits(), 255);
        assert_eq!(CurveType::Vesta.scalar_bits(), 255);
        assert_eq!(CurveType::Bn254.scalar_bits(), 254);
    }

    #[test]
    fn test_nova_error_display() {
        let err = NovaError::R1CSError("test error".to_string());
        assert!(err.to_string().contains("R1CS error"));
        
        let err = NovaError::FoldingError("fold failed".to_string());
        assert!(err.to_string().contains("Folding error"));
    }

    #[test]
    fn test_nova_metrics_default() {
        let metrics = NovaMetrics::default();
        
        assert_eq!(metrics.folding_steps, 0);
        assert_eq!(metrics.total_constraints, 0);
        assert_eq!(metrics.instances_folded, 0);
    }

    #[test]
    fn test_nova_metrics_display() {
        let metrics = NovaMetrics {
            folding_steps: 10,
            total_constraints: 1000,
            proof_generation_us: 5000,
            verification_us: 100,
            memory_bytes: 1024 * 1024,
            instances_folded: 10,
        };
        
        let display = format!("{}", metrics);
        assert!(display.contains("10"));
        assert!(display.contains("1000"));
    }
}

mod integration_tests {
    use super::*;

    #[test]
    fn test_full_ivc_workflow() {
        // 1. Create configuration
        let config = NovaConfig::default();
        
        // 2. Create prover
        let mut prover = NovaProver::new(config).unwrap();
        
        // 3. Define step circuit
        let circuit = TrivialCircuit::new(2);
        
        // 4. Setup public parameters
        let _params = prover.setup(&circuit).unwrap();
        
        // 5. Generate IVC proof
        let initial_state = vec![vec![1, 2, 3], vec![4, 5, 6]];
        let ivc_proof = prover.prove_ivc(&circuit, &initial_state, 5).unwrap();
        
        // 6. Verify IVC proof
        let is_valid = prover.verify_ivc(&circuit, &ivc_proof).unwrap();
        assert!(is_valid);
        
        // 7. Compress proof
        let compressed = prover.compress(&ivc_proof).unwrap();
        
        // 8. Verify compressed proof
        let is_compressed_valid = prover.verify_compressed(&compressed).unwrap();
        assert!(is_compressed_valid);
    }

    #[test]
    fn test_ivc_state_evolution() {
        let mut prover = NovaProver::new(NovaConfig::default()).unwrap();
        let circuit = TrivialCircuit::new(1);
        
        let initial_state = vec![vec![42, 0, 0, 0]];
        let proof = prover.prove_ivc(&circuit, &initial_state, 10).unwrap();
        
        // For trivial circuit, final state should equal initial state
        assert_eq!(proof.initial_state, proof.final_state);
    }

    #[test]
    fn test_multiple_proofs_same_circuit() {
        let mut prover = NovaProver::new(NovaConfig::default()).unwrap();
        let circuit = TrivialCircuit::new(1);
        
        // Generate multiple proofs
        let state1 = vec![vec![1]];
        let state2 = vec![vec![2]];
        let state3 = vec![vec![3]];
        
        let proof1 = prover.prove_ivc(&circuit, &state1, 3).unwrap();
        let proof2 = prover.prove_ivc(&circuit, &state2, 3).unwrap();
        let proof3 = prover.prove_ivc(&circuit, &state3, 3).unwrap();
        
        // All should verify
        assert!(prover.verify_ivc(&circuit, &proof1).unwrap());
        assert!(prover.verify_ivc(&circuit, &proof2).unwrap());
        assert!(prover.verify_ivc(&circuit, &proof3).unwrap());
    }

    #[test]
    fn test_different_step_counts() {
        let mut prover = NovaProver::new(NovaConfig::default()).unwrap();
        let circuit = TrivialCircuit::new(1);
        let initial_state = vec![vec![1]];
        
        // Test various step counts
        for num_steps in [1, 2, 5, 10, 20] {
            let proof = prover.prove_ivc(&circuit, &initial_state, num_steps).unwrap();
            assert_eq!(proof.num_steps, num_steps);
            assert!(prover.verify_ivc(&circuit, &proof).unwrap());
        }
    }
}

mod benchmark_tests {
    use super::*;
    use std::time::Instant;

    #[test]
    #[ignore] // Run with --ignored for benchmarks
    fn bench_ivc_proving_time() {
        let mut prover = NovaProver::new(NovaConfig::default()).unwrap();
        let circuit = TrivialCircuit::new(2);
        let initial_state = vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8]];
        
        // Warm up
        let _ = prover.prove_ivc(&circuit, &initial_state, 10);
        
        // Benchmark different step counts
        for num_steps in [10, 50, 100, 500] {
            let start = Instant::now();
            let proof = prover.prove_ivc(&circuit, &initial_state, num_steps).unwrap();
            let elapsed = start.elapsed();
            
            println!(
                "IVC {} steps: {:?} ({:.2} ms/step)",
                num_steps,
                elapsed,
                elapsed.as_millis() as f64 / num_steps as f64
            );
            
            assert_eq!(proof.num_steps, num_steps);
        }
    }

    #[test]
    #[ignore] // Run with --ignored for benchmarks
    fn bench_verification_time() {
        let mut prover = NovaProver::new(NovaConfig::default()).unwrap();
        let circuit = TrivialCircuit::new(1);
        let initial_state = vec![vec![1]];
        
        // Generate proof
        let proof = prover.prove_ivc(&circuit, &initial_state, 100).unwrap();
        
        // Benchmark verification
        let iterations = 100;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = prover.verify_ivc(&circuit, &proof).unwrap();
        }
        let elapsed = start.elapsed();
        
        println!(
            "Verification (100 steps, {} iterations): {:?} ({:.2} µs/verify)",
            iterations,
            elapsed,
            elapsed.as_micros() as f64 / iterations as f64
        );
    }
}
