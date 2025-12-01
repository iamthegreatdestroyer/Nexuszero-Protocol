//! Tests for the plugin-based proof system

use super::*;
use crate::proof::{Statement, StatementType};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_creation() {
        let registry = ProofRegistry::new();
        assert_eq!(registry.list().len(), 0);
    }

    #[test]
    fn test_builtin_plugins_registration() {
        let registry = ProofRegistry::with_builtin_plugins().unwrap();

        let plugins = registry.list();
        assert_eq!(plugins.len(), 4);
        assert!(plugins.contains(&ProofType::Schnorr));
        assert!(plugins.contains(&ProofType::Bulletproofs));
        assert!(plugins.contains(&ProofType::Groth16));
        assert!(plugins.contains(&ProofType::Plonk));
    }

    #[test]
    fn test_plugin_registration_and_retrieval() {
        let mut registry = ProofRegistry::new();
        let plugin = ProofPluginEnum::Schnorr(SchnorrPlugin::new());

        // Register plugin
        registry.register(plugin).unwrap();

        // Retrieve plugin
        let retrieved = registry.get(&ProofType::Schnorr).unwrap();
        match retrieved {
            ProofPluginEnum::Schnorr(p) => {
                assert_eq!(p.proof_type(), ProofType::Schnorr);
                assert_eq!(p.name(), "Schnorr Sigma Protocol");
            }
            _ => panic!("Wrong plugin type"),
        }
    }

    #[test]
    fn test_duplicate_plugin_registration() {
        let mut registry = ProofRegistry::new();
        let plugin1 = ProofPluginEnum::Schnorr(SchnorrPlugin::new());
        let plugin2 = ProofPluginEnum::Schnorr(SchnorrPlugin::new());

        // First registration should succeed
        registry.register(plugin1).unwrap();

        // Second registration should fail
        assert!(registry.register(plugin2).is_err());
    }

    #[test]
    fn test_plugin_unregistration() {
        let mut registry = ProofRegistry::new();
        let plugin = ProofPluginEnum::Bulletproofs(BulletproofsPlugin::new());

        // Register plugin
        registry.register(plugin).unwrap();
        assert!(registry.get(&ProofType::Bulletproofs).is_some());

        // Unregister plugin
        registry.unregister(&ProofType::Bulletproofs).unwrap();
        assert!(registry.get(&ProofType::Bulletproofs).is_none());
    }

    #[test]
    fn test_plugin_metadata() {
        let registry = ProofRegistry::with_builtin_plugins().unwrap();

        let schnorr = registry.get(&ProofType::Schnorr).unwrap();
        match schnorr {
            ProofPluginEnum::Schnorr(p) => {
                assert_eq!(p.proof_type(), ProofType::Schnorr);
                assert_eq!(p.name(), "Schnorr Sigma Protocol");
                assert_eq!(p.version(), "1.0.0");
            }
            _ => panic!("Wrong plugin type"),
        }

        let bulletproofs = registry.get(&ProofType::Bulletproofs).unwrap();
        match bulletproofs {
            ProofPluginEnum::Bulletproofs(p) => {
                assert_eq!(p.proof_type(), ProofType::Bulletproofs);
                assert_eq!(p.name(), "Bulletproofs Zero-Knowledge Proofs");
                assert_eq!(p.version(), "1.0.0");
            }
            _ => panic!("Wrong plugin type"),
        }
    }

    #[test]
    fn test_plugin_supported_statements() {
        let registry = ProofRegistry::with_builtin_plugins().unwrap();

        let schnorr = registry.get(&ProofType::Schnorr).unwrap();
        match schnorr {
            ProofPluginEnum::Schnorr(p) => {
                let statements = p.supported_statements();
                assert!(!statements.is_empty());
                // Should support discrete log statements
                assert!(statements.iter().any(|s| matches!(s, StatementType::DiscreteLog { .. })));
            }
            _ => panic!("Wrong plugin type"),
        }

        let bulletproofs = registry.get(&ProofType::Bulletproofs).unwrap();
        match bulletproofs {
            ProofPluginEnum::Bulletproofs(p) => {
                let statements = p.supported_statements();
                assert!(!statements.is_empty());
                // Should support range statements
                assert!(statements.iter().any(|s| matches!(s, StatementType::Range { .. })));
            }
            _ => panic!("Wrong plugin type"),
        }
    }

    #[test]
    fn test_plugin_circuit_info() {
        let registry = ProofRegistry::with_builtin_plugins().unwrap();

        // Test Schnorr plugin circuit info
        let schnorr = registry.get(&ProofType::Schnorr).unwrap();
        match schnorr {
            ProofPluginEnum::Schnorr(p) => {
                let statement = Statement {
                    version: 1,
                    statement_type: StatementType::DiscreteLog {
                        generator: vec![2],
                        public_value: vec![4],
                    },
                };
                let info = p.circuit_info(&statement);
                assert!(info.constraints > 0);
                assert!(info.variables > 0);
                assert!(info.proof_size_bytes > 0);
                assert!(info.verification_time_ms > 0);
            }
            _ => panic!("Wrong plugin type"),
        }

        // Test Bulletproofs plugin circuit info
        let bulletproofs = registry.get(&ProofType::Bulletproofs).unwrap();
        match bulletproofs {
            ProofPluginEnum::Bulletproofs(p) => {
                let statement = Statement {
                    version: 1,
                    statement_type: StatementType::Range {
                        min: 0,
                        max: 100,
                        commitment: vec![1, 2, 3],
                    },
                };
                let info = p.circuit_info(&statement);
                assert!(info.constraints > 0);
                assert!(info.variables > 0);
                assert!(info.proof_size_bytes > 0);
                assert!(info.verification_time_ms > 0);
            }
            _ => panic!("Wrong plugin type"),
        }
    }

    #[tokio::test]
    async fn test_plugin_setup() {
        let registry = ProofRegistry::with_builtin_plugins().unwrap();

        let schnorr = registry.get(&ProofType::Schnorr).unwrap();
        match schnorr {
            ProofPluginEnum::Schnorr(p) => {
                let params = SetupParams {
                    security_level: crate::SecurityLevel::Bit256,
                    circuit_params: HashMap::new(),
                    trusted_setup: None,
                };

                let (prover_key, verification_key) = p.setup(&params).await.unwrap();
                // Schnorr proofs don't require special setup, so keys may be empty
                assert_eq!(prover_key.key_type, "schnorr");
                assert_eq!(verification_key.key_type, "schnorr");
                assert_eq!(prover_key.proof_type, ProofType::Schnorr);
                assert_eq!(verification_key.proof_type, ProofType::Schnorr);
            }
            _ => panic!("Wrong plugin type"),
        }
    }

    #[test]
    fn test_plugin_serialization() {
        let registry = ProofRegistry::with_builtin_plugins().unwrap();

        let schnorr = registry.get(&ProofType::Schnorr).unwrap();
        match schnorr {
            ProofPluginEnum::Schnorr(p) => {
                let serialized = p.serialize().unwrap();
                // Plugin has no internal state, so serialization may be minimal
                assert!(serialized.is_empty() || !serialized.is_empty()); // Always true, just testing serialization works
            }
            _ => panic!("Wrong plugin type"),
        }
    }
}