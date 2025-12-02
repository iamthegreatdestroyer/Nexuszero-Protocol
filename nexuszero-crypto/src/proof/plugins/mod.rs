//! Plugin-based proof system architecture
//!
//! This module provides a plugin-based architecture for zero-knowledge proof systems,
//! allowing dynamic registration and use of different proof implementations while
//! maintaining backward compatibility with existing APIs.

use crate::proof::{Statement, Witness, Proof};
use crate::{CryptoError, CryptoResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// Plugin implementations
pub mod schnorr;
pub mod bulletproofs;
pub mod groth16;
pub mod plonk;
pub mod stark;
#[cfg(test)]
pub mod tests;

pub use schnorr::SchnorrPlugin;
pub use bulletproofs::BulletproofsPlugin;
pub use groth16::Groth16Plugin;
pub use plonk::PlonkPlugin;
pub use stark::StarkPlugin;

/// Unique identifier for a proof system
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProofType {
    /// Schnorr-style sigma protocol proofs
    Schnorr,
    /// Bulletproofs range proofs
    Bulletproofs,
    /// Groth16 ZK-SNARK
    Groth16,
    /// Enhanced Groth16 with additional constraints
    Groth16Plus,
    /// Plonk proof system
    Plonk,
    /// STARK proof
    Stark,
    /// Custom proof type with identifier
    Custom(String),
}

impl std::fmt::Display for ProofType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProofType::Schnorr => write!(f, "Schnorr"),
            ProofType::Bulletproofs => write!(f, "Bulletproofs"),
            ProofType::Groth16 => write!(f, "Groth16"),
            ProofType::Groth16Plus => write!(f, "Groth16Plus"),
            ProofType::Plonk => write!(f, "Plonk"),
            ProofType::Stark => write!(f, "Stark"),
            ProofType::Custom(name) => write!(f, "Custom({})", name),
        }
    }
}

/// Setup parameters for proof system initialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SetupParams {
    /// Security level for the proof system
    pub security_level: crate::SecurityLevel,
    /// Circuit-specific parameters
    pub circuit_params: HashMap<String, serde_json::Value>,
    /// Trusted setup data (if required)
    pub trusted_setup: Option<Vec<u8>>,
}

/// Verification key for a proof system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationKey {
    /// Key data
    pub data: Vec<u8>,
    /// Key type identifier
    pub key_type: String,
    /// Associated proof type
    pub proof_type: ProofType,
}

/// Prover key for a proof system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProverKey {
    /// Key data
    pub data: Vec<u8>,
    /// Key type identifier
    pub key_type: String,
    /// Associated proof type
    pub proof_type: ProofType,
}

/// Enum containing all available proof plugins
#[derive(Debug, Clone)]
pub enum ProofPluginEnum {
    Schnorr(SchnorrPlugin),
    Bulletproofs(BulletproofsPlugin),
    Groth16(Groth16Plugin),
    Plonk(PlonkPlugin),
    Stark(StarkPlugin),
}

impl ProofPluginEnum {
    /// Get the proof type for this plugin
    pub fn proof_type(&self) -> ProofType {
        match self {
            ProofPluginEnum::Schnorr(p) => p.proof_type(),
            ProofPluginEnum::Bulletproofs(p) => p.proof_type(),
            ProofPluginEnum::Groth16(p) => p.proof_type(),
            ProofPluginEnum::Plonk(p) => p.proof_type(),
            ProofPluginEnum::Stark(p) => p.proof_type(),
        }
    }

    /// Get the name for this plugin
    pub fn name(&self) -> &'static str {
        match self {
            ProofPluginEnum::Schnorr(p) => p.name(),
            ProofPluginEnum::Bulletproofs(p) => p.name(),
            ProofPluginEnum::Groth16(p) => p.name(),
            ProofPluginEnum::Plonk(p) => p.name(),
            ProofPluginEnum::Stark(p) => p.name(),
        }
    }

    /// Get the version for this plugin
    pub fn version(&self) -> &'static str {
        match self {
            ProofPluginEnum::Schnorr(p) => p.version(),
            ProofPluginEnum::Bulletproofs(p) => p.version(),
            ProofPluginEnum::Groth16(p) => p.version(),
            ProofPluginEnum::Plonk(p) => p.version(),
            ProofPluginEnum::Stark(p) => p.version(),
        }
    }

    /// Get supported statements for this plugin
    pub fn supported_statements(&self) -> Vec<crate::proof::statement::StatementType> {
        match self {
            ProofPluginEnum::Schnorr(p) => p.supported_statements(),
            ProofPluginEnum::Bulletproofs(p) => p.supported_statements(),
            ProofPluginEnum::Groth16(p) => p.supported_statements(),
            ProofPluginEnum::Plonk(p) => p.supported_statements(),
            ProofPluginEnum::Stark(p) => p.supported_statements(),
        }
    }

    /// Setup the proof system
    pub async fn setup(&self, params: &SetupParams) -> CryptoResult<(ProverKey, VerificationKey)> {
        match self {
            ProofPluginEnum::Schnorr(p) => p.setup(params).await,
            ProofPluginEnum::Bulletproofs(p) => p.setup(params).await,
            ProofPluginEnum::Groth16(p) => p.setup(params).await,
            ProofPluginEnum::Plonk(p) => p.setup(params).await,
            ProofPluginEnum::Stark(p) => p.setup(params).await,
        }
    }

    /// Generate a proof
    pub async fn prove(&self, statement: &Statement, witness: &Witness, prover_key: &ProverKey) -> CryptoResult<Proof> {
        match self {
            ProofPluginEnum::Schnorr(p) => p.prove(statement, witness, prover_key).await,
            ProofPluginEnum::Bulletproofs(p) => p.prove(statement, witness, prover_key).await,
            ProofPluginEnum::Groth16(p) => p.prove(statement, witness, prover_key).await,
            ProofPluginEnum::Plonk(p) => p.prove(statement, witness, prover_key).await,
            ProofPluginEnum::Stark(p) => p.prove(statement, witness, prover_key).await,
        }
    }

    /// Verify a proof
    pub async fn verify(&self, statement: &Statement, proof: &Proof, verification_key: &VerificationKey) -> CryptoResult<bool> {
        match self {
            ProofPluginEnum::Schnorr(p) => p.verify(statement, proof, verification_key).await,
            ProofPluginEnum::Bulletproofs(p) => p.verify(statement, proof, verification_key).await,
            ProofPluginEnum::Groth16(p) => p.verify(statement, proof, verification_key).await,
            ProofPluginEnum::Plonk(p) => p.verify(statement, proof, verification_key).await,
            ProofPluginEnum::Stark(p) => p.verify(statement, proof, verification_key).await,
        }
    }

    /// Serialize the plugin
    pub fn serialize(&self) -> CryptoResult<Vec<u8>> {
        match self {
            ProofPluginEnum::Schnorr(p) => p.serialize(),
            ProofPluginEnum::Bulletproofs(p) => p.serialize(),
            ProofPluginEnum::Groth16(p) => p.serialize(),
            ProofPluginEnum::Plonk(p) => p.serialize(),
            ProofPluginEnum::Stark(p) => p.serialize(),
        }
    }

    /// Get circuit info
    pub fn circuit_info(&self, statement: &Statement) -> CircuitInfo {
        match self {
            ProofPluginEnum::Schnorr(p) => p.circuit_info(statement),
            ProofPluginEnum::Bulletproofs(p) => p.circuit_info(statement),
            ProofPluginEnum::Groth16(p) => p.circuit_info(statement),
            ProofPluginEnum::Plonk(p) => p.circuit_info(statement),
            ProofPluginEnum::Stark(p) => p.circuit_info(statement),
        }
    }
}

/// Circuit complexity and size information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitInfo {
    /// Number of constraints in the circuit
    pub constraints: usize,
    /// Number of variables in the circuit
    pub variables: usize,
    /// Estimated proof size in bytes
    pub proof_size_bytes: usize,
    /// Estimated verification time in milliseconds
    pub verification_time_ms: u64,
    /// Circuit-specific metadata
    pub metadata: HashMap<String, String>,
}

/// Registry for managing proof system plugins
pub struct ProofRegistry {
    plugins: HashMap<ProofType, ProofPluginEnum>,
}

impl ProofRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            plugins: HashMap::new(),
        }
    }

    /// Register a proof plugin
    pub fn register(&mut self, plugin: ProofPluginEnum) -> CryptoResult<()> {
        let proof_type = plugin.proof_type();
        if self.plugins.contains_key(&proof_type) {
            return Err(CryptoError::ProofError(
                format!("Plugin for proof type '{}' is already registered", proof_type)
            ));
        }
        self.plugins.insert(proof_type, plugin);
        Ok(())
    }

    /// Get a plugin by proof type
    pub fn get(&self, proof_type: &ProofType) -> Option<&ProofPluginEnum> {
        self.plugins.get(proof_type)
    }

    /// Unregister a plugin
    pub fn unregister(&mut self, proof_type: &ProofType) -> CryptoResult<()> {
        if self.plugins.remove(proof_type).is_none() {
            return Err(CryptoError::ProofError(
                format!("Plugin for proof type '{}' is not registered", proof_type)
            ));
        }
        Ok(())
    }

    /// List all registered proof types
    pub fn list(&self) -> Vec<ProofType> {
        self.plugins.keys().cloned().collect()
    }

    /// Check if a proof type is supported
    pub fn is_supported(&self, proof_type: &ProofType) -> bool {
        self.plugins.contains_key(proof_type)
    }

    /// Get plugin information for all registered plugins
    pub fn plugin_info(&self) -> Vec<PluginInfo> {
        self.plugins.values().map(|plugin| PluginInfo {
            proof_type: plugin.proof_type(),
            name: plugin.name().to_string(),
            version: plugin.version().to_string(),
            supported_statements: plugin.supported_statements(),
        }).collect()
    }
}

/// Information about a registered plugin
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginInfo {
    /// Proof type
    pub proof_type: ProofType,
    /// Plugin name
    pub name: String,
    /// Plugin version
    pub version: String,
    /// Supported statement types
    pub supported_statements: Vec<crate::proof::statement::StatementType>,
}

impl Default for ProofRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ProofRegistry {
    /// Create a new registry with all built-in plugins registered
    pub fn with_builtin_plugins() -> CryptoResult<Self> {
        let mut registry = Self::new();

        // Register all built-in plugins
        registry.register(ProofPluginEnum::Schnorr(SchnorrPlugin::new()))?;
        registry.register(ProofPluginEnum::Bulletproofs(BulletproofsPlugin::new()))?;
        registry.register(ProofPluginEnum::Groth16(Groth16Plugin::new()))?;
        registry.register(ProofPluginEnum::Plonk(PlonkPlugin::new()))?;

        Ok(registry)
    }
}

/// Base trait for circuit components
pub trait CircuitComponent: Send + Sync {
    /// Get the component type identifier
    fn component_type(&self) -> &'static str;

    /// Get the number of constraints this component adds
    fn constraints(&self) -> usize;

    /// Get the number of variables this component uses
    fn variables(&self) -> usize;

    /// Get public inputs required by this component
    fn public_inputs(&self) -> Vec<String>;

    /// Get private inputs required by this component
    fn private_inputs(&self) -> Vec<String>;

    /// Validate that the component is properly configured
    fn validate(&self) -> CryptoResult<()>;

    /// Get component metadata
    fn metadata(&self) -> HashMap<String, String>;
}

