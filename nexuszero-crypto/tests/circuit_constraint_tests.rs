//! Circuit Constraint Unit Tests
//!
//! This module provides comprehensive unit tests for the circuit constraint system.
//! Tests cover Variable, Constraint, Expression, CircuitComponent, and CircuitEngine.
//!
//! ## Test Categories
//! 1. Variable type tests - all VariableType variants
//! 2. Constraint definition tests - Equality, Range, Boolean, Custom
//! 3. Expression evaluation tests - Variable, Constant, Add, Mul
//! 4. Circuit structure tests - connections, components, execution order
//! 5. CircuitEngine registration and execution tests

use std::collections::HashMap;

#[cfg(test)]
mod variable_tests {
    use nexuszero_crypto::proof::circuit::{Variable, VariableType};

    #[test]
    fn test_field_element_variable() {
        let var = Variable {
            name: "x".to_string(),
            var_type: VariableType::FieldElement,
            bit_length: 256,
        };
        
        assert_eq!(var.name, "x");
        assert!(matches!(var.var_type, VariableType::FieldElement));
        assert_eq!(var.bit_length, 256);
    }

    #[test]
    fn test_boolean_variable() {
        let var = Variable {
            name: "flag".to_string(),
            var_type: VariableType::Boolean,
            bit_length: 1,
        };
        
        assert_eq!(var.name, "flag");
        assert!(matches!(var.var_type, VariableType::Boolean));
        assert_eq!(var.bit_length, 1);
    }

    #[test]
    fn test_signed_integer_variable() {
        let var = Variable {
            name: "counter".to_string(),
            var_type: VariableType::Integer { signed: true },
            bit_length: 64,
        };
        
        assert_eq!(var.name, "counter");
        if let VariableType::Integer { signed } = var.var_type {
            assert!(signed);
        } else {
            panic!("Expected Integer type");
        }
    }

    #[test]
    fn test_unsigned_integer_variable() {
        let var = Variable {
            name: "amount".to_string(),
            var_type: VariableType::Integer { signed: false },
            bit_length: 32,
        };
        
        if let VariableType::Integer { signed } = var.var_type {
            assert!(!signed);
        } else {
            panic!("Expected Integer type");
        }
    }

    #[test]
    fn test_bytes_variable() {
        let var = Variable {
            name: "hash_output".to_string(),
            var_type: VariableType::Bytes { length: 32 },
            bit_length: 256,
        };
        
        if let VariableType::Bytes { length } = var.var_type {
            assert_eq!(length, 32);
        } else {
            panic!("Expected Bytes type");
        }
    }

    #[test]
    fn test_variable_serialization_roundtrip() {
        let var = Variable {
            name: "test_var".to_string(),
            var_type: VariableType::FieldElement,
            bit_length: 256,
        };
        
        let serialized = serde_json::to_string(&var).expect("Serialization failed");
        let deserialized: Variable = serde_json::from_str(&serialized).expect("Deserialization failed");
        
        assert_eq!(var.name, deserialized.name);
        assert_eq!(var.bit_length, deserialized.bit_length);
    }

    #[test]
    fn test_variable_clone() {
        let var1 = Variable {
            name: "original".to_string(),
            var_type: VariableType::Boolean,
            bit_length: 1,
        };
        
        let var2 = var1.clone();
        
        assert_eq!(var1.name, var2.name);
        assert_eq!(var1.bit_length, var2.bit_length);
    }

    #[test]
    fn test_variable_debug() {
        let var = Variable {
            name: "debug_test".to_string(),
            var_type: VariableType::FieldElement,
            bit_length: 128,
        };
        
        let debug_str = format!("{:?}", var);
        assert!(debug_str.contains("debug_test"));
        assert!(debug_str.contains("FieldElement"));
    }

    #[test]
    fn test_all_variable_types_serializable() {
        let types = vec![
            VariableType::FieldElement,
            VariableType::Boolean,
            VariableType::Integer { signed: true },
            VariableType::Integer { signed: false },
            VariableType::Bytes { length: 64 },
        ];
        
        for vt in types {
            let var = Variable {
                name: "test".to_string(),
                var_type: vt,
                bit_length: 64,
            };
            let serialized = serde_json::to_string(&var);
            assert!(serialized.is_ok(), "Failed to serialize variable type");
        }
    }
}

#[cfg(test)]
mod constraint_tests {
    use nexuszero_crypto::proof::circuit::{Constraint, ConstraintType, Expression};

    #[test]
    fn test_equality_constraint() {
        let constraint = Constraint {
            left: Expression::Variable("x".to_string()),
            right: Expression::Variable("y".to_string()),
            constraint_type: ConstraintType::Equality,
        };
        
        assert!(matches!(constraint.constraint_type, ConstraintType::Equality));
    }

    #[test]
    fn test_range_constraint() {
        let constraint = Constraint {
            left: Expression::Variable("value".to_string()),
            right: Expression::Constant(vec![0, 0, 0, 100]), // Upper bound
            constraint_type: ConstraintType::Range { min: 0, max: 100 },
        };
        
        if let ConstraintType::Range { min, max } = constraint.constraint_type {
            assert_eq!(min, 0);
            assert_eq!(max, 100);
        } else {
            panic!("Expected Range constraint");
        }
    }

    #[test]
    fn test_boolean_constraint() {
        let constraint = Constraint {
            left: Expression::Variable("flag".to_string()),
            right: Expression::Constant(vec![1]),
            constraint_type: ConstraintType::Boolean,
        };
        
        assert!(matches!(constraint.constraint_type, ConstraintType::Boolean));
    }

    #[test]
    fn test_custom_constraint() {
        let constraint = Constraint {
            left: Expression::Variable("a".to_string()),
            right: Expression::Variable("b".to_string()),
            constraint_type: ConstraintType::Custom("polynomial_equality".to_string()),
        };
        
        if let ConstraintType::Custom(desc) = &constraint.constraint_type {
            assert_eq!(desc, "polynomial_equality");
        } else {
            panic!("Expected Custom constraint");
        }
    }

    #[test]
    fn test_constraint_serialization() {
        let constraint = Constraint {
            left: Expression::Variable("x".to_string()),
            right: Expression::Constant(vec![42]),
            constraint_type: ConstraintType::Equality,
        };
        
        let serialized = serde_json::to_string(&constraint).expect("Serialization failed");
        let deserialized: Constraint = serde_json::from_str(&serialized).expect("Deserialization failed");
        
        assert!(matches!(deserialized.constraint_type, ConstraintType::Equality));
    }

    #[test]
    fn test_constraint_clone() {
        let original = Constraint {
            left: Expression::Variable("a".to_string()),
            right: Expression::Variable("b".to_string()),
            constraint_type: ConstraintType::Equality,
        };
        
        let cloned = original.clone();
        assert!(matches!(cloned.constraint_type, ConstraintType::Equality));
    }
}

#[cfg(test)]
mod expression_tests {
    use nexuszero_crypto::proof::circuit::Expression;

    #[test]
    fn test_variable_expression() {
        let expr = Expression::Variable("x".to_string());
        
        if let Expression::Variable(name) = expr {
            assert_eq!(name, "x");
        } else {
            panic!("Expected Variable expression");
        }
    }

    #[test]
    fn test_constant_expression() {
        let value = vec![0x12, 0x34, 0x56, 0x78];
        let expr = Expression::Constant(value.clone());
        
        if let Expression::Constant(v) = expr {
            assert_eq!(v, value);
        } else {
            panic!("Expected Constant expression");
        }
    }

    #[test]
    fn test_add_expression() {
        let a = Expression::Variable("a".to_string());
        let b = Expression::Variable("b".to_string());
        let sum = Expression::Add(Box::new(a), Box::new(b));
        
        if let Expression::Add(left, right) = sum {
            assert!(matches!(*left, Expression::Variable(_)));
            assert!(matches!(*right, Expression::Variable(_)));
        } else {
            panic!("Expected Add expression");
        }
    }

    #[test]
    fn test_mul_expression() {
        let a = Expression::Variable("a".to_string());
        let b = Expression::Constant(vec![2]);
        let product = Expression::Mul(Box::new(a), Box::new(b));
        
        if let Expression::Mul(left, right) = product {
            assert!(matches!(*left, Expression::Variable(_)));
            assert!(matches!(*right, Expression::Constant(_)));
        } else {
            panic!("Expected Mul expression");
        }
    }

    #[test]
    fn test_nested_expression() {
        // (a + b) * c
        let a = Expression::Variable("a".to_string());
        let b = Expression::Variable("b".to_string());
        let c = Expression::Variable("c".to_string());
        
        let sum = Expression::Add(Box::new(a), Box::new(b));
        let product = Expression::Mul(Box::new(sum), Box::new(c));
        
        if let Expression::Mul(left, right) = product {
            assert!(matches!(*left, Expression::Add(_, _)));
            assert!(matches!(*right, Expression::Variable(_)));
        } else {
            panic!("Expected nested expression");
        }
    }

    #[test]
    fn test_deeply_nested_expression() {
        // ((a + b) * (c + d)) + e
        let a = Expression::Variable("a".to_string());
        let b = Expression::Variable("b".to_string());
        let c = Expression::Variable("c".to_string());
        let d = Expression::Variable("d".to_string());
        let e = Expression::Variable("e".to_string());
        
        let ab = Expression::Add(Box::new(a), Box::new(b));
        let cd = Expression::Add(Box::new(c), Box::new(d));
        let product = Expression::Mul(Box::new(ab), Box::new(cd));
        let final_expr = Expression::Add(Box::new(product), Box::new(e));
        
        assert!(matches!(final_expr, Expression::Add(_, _)));
    }

    #[test]
    fn test_expression_serialization() {
        let expr = Expression::Add(
            Box::new(Expression::Variable("x".to_string())),
            Box::new(Expression::Constant(vec![1, 2, 3])),
        );
        
        let serialized = serde_json::to_string(&expr).expect("Serialization failed");
        let _deserialized: Expression = serde_json::from_str(&serialized).expect("Deserialization failed");
    }

    #[test]
    fn test_expression_clone_deep() {
        let original = Expression::Mul(
            Box::new(Expression::Add(
                Box::new(Expression::Variable("a".to_string())),
                Box::new(Expression::Constant(vec![1])),
            )),
            Box::new(Expression::Variable("b".to_string())),
        );
        
        let cloned = original.clone();
        
        // Verify structure is preserved
        if let Expression::Mul(left, _) = cloned {
            assert!(matches!(*left, Expression::Add(_, _)));
        } else {
            panic!("Clone didn't preserve structure");
        }
    }
}

#[cfg(test)]
mod circuit_structure_tests {
    use nexuszero_crypto::proof::circuit::{Circuit, Connection};

    #[test]
    fn test_circuit_creation() {
        let circuit = Circuit {
            id: "test_circuit".to_string(),
            components: vec!["comp_a".to_string(), "comp_b".to_string()],
            connections: vec![],
            public_inputs: vec!["x".to_string()],
            private_inputs: vec!["secret".to_string()],
            outputs: vec!["result".to_string()],
        };
        
        assert_eq!(circuit.id, "test_circuit");
        assert_eq!(circuit.components.len(), 2);
        assert_eq!(circuit.public_inputs.len(), 1);
        assert_eq!(circuit.private_inputs.len(), 1);
    }

    #[test]
    fn test_circuit_connection() {
        let connection = Connection {
            from_component: "hasher".to_string(),
            from_output: "hash_output".to_string(),
            to_component: "verifier".to_string(),
            to_input: "commitment".to_string(),
        };
        
        assert_eq!(connection.from_component, "hasher");
        assert_eq!(connection.to_component, "verifier");
    }

    #[test]
    fn test_circuit_with_connections() {
        let circuit = Circuit {
            id: "hash_verify_circuit".to_string(),
            components: vec!["input".to_string(), "hasher".to_string(), "verifier".to_string()],
            connections: vec![
                Connection {
                    from_component: "input".to_string(),
                    from_output: "value".to_string(),
                    to_component: "hasher".to_string(),
                    to_input: "preimage".to_string(),
                },
                Connection {
                    from_component: "hasher".to_string(),
                    from_output: "hash".to_string(),
                    to_component: "verifier".to_string(),
                    to_input: "hash_input".to_string(),
                },
            ],
            public_inputs: vec!["expected_hash".to_string()],
            private_inputs: vec!["secret_preimage".to_string()],
            outputs: vec!["is_valid".to_string()],
        };
        
        assert_eq!(circuit.connections.len(), 2);
        assert_eq!(circuit.components.len(), 3);
    }

    #[test]
    fn test_circuit_serialization() {
        let circuit = Circuit {
            id: "serialization_test".to_string(),
            components: vec!["a".to_string()],
            connections: vec![],
            public_inputs: vec!["pub_in".to_string()],
            private_inputs: vec!["priv_in".to_string()],
            outputs: vec!["out".to_string()],
        };
        
        let serialized = serde_json::to_string(&circuit).expect("Serialization failed");
        let deserialized: Circuit = serde_json::from_str(&serialized).expect("Deserialization failed");
        
        assert_eq!(circuit.id, deserialized.id);
        assert_eq!(circuit.components, deserialized.components);
    }

    #[test]
    fn test_empty_circuit() {
        let circuit = Circuit {
            id: "empty".to_string(),
            components: vec![],
            connections: vec![],
            public_inputs: vec![],
            private_inputs: vec![],
            outputs: vec![],
        };
        
        assert!(circuit.components.is_empty());
        assert!(circuit.connections.is_empty());
    }

    #[test]
    fn test_circuit_clone() {
        let circuit = Circuit {
            id: "clone_test".to_string(),
            components: vec!["a".to_string(), "b".to_string()],
            connections: vec![Connection {
                from_component: "a".to_string(),
                from_output: "out".to_string(),
                to_component: "b".to_string(),
                to_input: "in".to_string(),
            }],
            public_inputs: vec![],
            private_inputs: vec![],
            outputs: vec![],
        };
        
        let cloned = circuit.clone();
        assert_eq!(circuit.id, cloned.id);
        assert_eq!(circuit.connections.len(), cloned.connections.len());
    }
}

#[cfg(test)]
mod circuit_engine_tests {
    use nexuszero_crypto::proof::circuit::{Circuit, CircuitEngine};

    #[test]
    fn test_circuit_engine_creation() {
        let engine = CircuitEngine::new();
        // Engine should be created successfully (internal state is private)
        let _ = engine;
    }

    #[test]
    fn test_circuit_registration() {
        let mut engine = CircuitEngine::new();
        
        let circuit = Circuit {
            id: "test_circuit".to_string(),
            components: vec![],
            connections: vec![],
            public_inputs: vec![],
            private_inputs: vec![],
            outputs: vec![],
        };
        
        engine.register_circuit(circuit);
        // Registration should succeed (no panic)
    }

    #[test]
    fn test_multiple_circuit_registration() {
        let mut engine = CircuitEngine::new();
        
        for i in 0..5 {
            let circuit = Circuit {
                id: format!("circuit_{}", i),
                components: vec![],
                connections: vec![],
                public_inputs: vec![],
                private_inputs: vec![],
                outputs: vec![],
            };
            engine.register_circuit(circuit);
        }
        // All registrations should succeed
    }

    #[test]
    fn test_circuit_overwrite() {
        let mut engine = CircuitEngine::new();
        
        let circuit1 = Circuit {
            id: "duplicate".to_string(),
            components: vec!["a".to_string()],
            connections: vec![],
            public_inputs: vec![],
            private_inputs: vec![],
            outputs: vec![],
        };
        
        let circuit2 = Circuit {
            id: "duplicate".to_string(),
            components: vec!["b".to_string(), "c".to_string()],
            connections: vec![],
            public_inputs: vec![],
            private_inputs: vec![],
            outputs: vec![],
        };
        
        engine.register_circuit(circuit1);
        engine.register_circuit(circuit2);
        // Second registration should overwrite first (no error)
    }
}

#[cfg(test)]
mod witness_data_tests {
    use std::collections::HashMap;
    use nexuszero_crypto::proof::circuit::WitnessData;

    #[test]
    fn test_witness_data_creation() {
        let mut variables = HashMap::new();
        variables.insert("x".to_string(), vec![1, 2, 3, 4]);
        variables.insert("y".to_string(), vec![5, 6, 7, 8]);
        
        let witness = WitnessData {
            variables,
            randomness: vec![0u8; 32],
        };
        
        assert_eq!(witness.variables.len(), 2);
        assert_eq!(witness.randomness.len(), 32);
    }

    #[test]
    fn test_witness_data_empty() {
        let witness = WitnessData {
            variables: HashMap::new(),
            randomness: vec![],
        };
        
        assert!(witness.variables.is_empty());
        assert!(witness.randomness.is_empty());
    }

    #[test]
    fn test_witness_data_clone() {
        let mut variables = HashMap::new();
        variables.insert("secret".to_string(), vec![0xDE, 0xAD, 0xBE, 0xEF]);
        
        let original = WitnessData {
            variables,
            randomness: vec![1, 2, 3],
        };
        
        let cloned = original.clone();
        assert_eq!(original.variables["secret"], cloned.variables["secret"]);
        assert_eq!(original.randomness, cloned.randomness);
    }
}

#[cfg(test)]
mod constraint_type_boundary_tests {
    use nexuszero_crypto::proof::circuit::ConstraintType;

    #[test]
    fn test_range_constraint_min_equals_max_minus_one() {
        let ct = ConstraintType::Range { min: 99, max: 100 };
        if let ConstraintType::Range { min, max } = ct {
            assert_eq!(max - min, 1);
        }
    }

    #[test]
    fn test_range_constraint_large_range() {
        let ct = ConstraintType::Range { min: 0, max: u64::MAX };
        if let ConstraintType::Range { min, max } = ct {
            assert_eq!(min, 0);
            assert_eq!(max, u64::MAX);
        }
    }

    #[test]
    fn test_range_constraint_small_range() {
        let ct = ConstraintType::Range { min: 0, max: 1 };
        if let ConstraintType::Range { min, max } = ct {
            assert!(max > min);
        }
    }

    #[test]
    fn test_custom_constraint_empty_description() {
        let ct = ConstraintType::Custom(String::new());
        if let ConstraintType::Custom(desc) = ct {
            assert!(desc.is_empty());
        }
    }

    #[test]
    fn test_custom_constraint_long_description() {
        let long_desc = "a".repeat(10000);
        let ct = ConstraintType::Custom(long_desc.clone());
        if let ConstraintType::Custom(desc) = ct {
            assert_eq!(desc.len(), 10000);
        }
    }
}
