//! Privacy level management and Adaptive Privacy Morphing (APM) engine

use serde::{Deserialize, Serialize};

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

use crate::error::{NexusZeroError, Result};

/// The 6-level privacy spectrum
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[repr(u8)]
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub enum PrivacyLevel {
    /// Public blockchain parity
    Transparent = 0,
    /// Address obfuscation
    Pseudonymous = 1,
    /// Encrypted amounts
    Confidential = 2,
    /// Full transaction privacy
    Private = 3,
    /// Unlinkable transactions
    Anonymous = 4,
    /// Maximum privacy, ZK everything
    Sovereign = 5,
}

impl PrivacyLevel {
    /// Get privacy level from integer value
    pub fn from_u8(value: u8) -> Result<Self> {
        match value {
            0 => Ok(PrivacyLevel::Transparent),
            1 => Ok(PrivacyLevel::Pseudonymous),
            2 => Ok(PrivacyLevel::Confidential),
            3 => Ok(PrivacyLevel::Private),
            4 => Ok(PrivacyLevel::Anonymous),
            5 => Ok(PrivacyLevel::Sovereign),
            _ => Err(NexusZeroError::InvalidPrivacyLevel(value)),
        }
    }

    /// Get human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            PrivacyLevel::Transparent => "Public blockchain parity, no privacy enhancements",
            PrivacyLevel::Pseudonymous => "Address obfuscation with basic decoys",
            PrivacyLevel::Confidential => "Encrypted amounts, visible addresses",
            PrivacyLevel::Private => "Full transaction privacy with ZK proofs",
            PrivacyLevel::Anonymous => "Unlinkable transactions, large anonymity set",
            PrivacyLevel::Sovereign => "Maximum privacy, quantum-resistant ZK proofs",
        }
    }

    /// Get security level in bits
    pub fn security_bits(&self) -> u32 {
        match self {
            PrivacyLevel::Transparent => 0,
            PrivacyLevel::Pseudonymous => 80,
            PrivacyLevel::Confidential => 128,
            PrivacyLevel::Private => 192,
            PrivacyLevel::Anonymous => 256,
            PrivacyLevel::Sovereign => 256,
        }
    }

    /// Estimated proof generation time in milliseconds
    pub fn estimated_proof_time_ms(&self) -> u32 {
        match self {
            PrivacyLevel::Transparent => 0,
            PrivacyLevel::Pseudonymous => 50,
            PrivacyLevel::Confidential => 100,
            PrivacyLevel::Private => 250,
            PrivacyLevel::Anonymous => 500,
            PrivacyLevel::Sovereign => 1000,
        }
    }
}

/// Configuration parameters for a privacy level
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "wasm", wasm_bindgen(getter_with_clone))]
pub struct PrivacyParameters {
    pub lattice_n: u32,
    pub modulus_q: u64,
    pub sigma: f64,
    pub security_bits: u32,
    pub proof_strategy: String,
    pub anonymity_set_size: Option<u32>,
    pub decoy_count: Option<u32>,
}

/// Transaction context for privacy recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionContext {
    pub value_usd: f64,
    pub requires_compliance: bool,
    pub preferred_level: Option<u8>,
    pub risk_score: f64,
    pub jurisdiction: String,
    pub counterparty_known: bool,
}

impl Default for TransactionContext {
    fn default() -> Self {
        Self {
            value_usd: 0.0,
            requires_compliance: false,
            preferred_level: None,
            risk_score: 0.0,
            jurisdiction: "US".to_string(),
            counterparty_known: false,
        }
    }
}

/// Privacy recommendation from APM engine
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "wasm", wasm_bindgen(getter_with_clone))]
pub struct PrivacyRecommendation {
    pub level: u8,
    pub reasons: Vec<String>,
    pub estimated_proof_time_ms: u32,
    pub estimated_cost_gas: u64,
}

/// Adaptive Privacy Morphing (APM) Engine
#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub struct PrivacyEngine {
    parameters: std::collections::HashMap<u8, PrivacyParameters>,
    gas_costs: std::collections::HashMap<u8, u64>,
}

#[cfg_attr(feature = "wasm", wasm_bindgen)]
impl PrivacyEngine {
    /// Create a new privacy engine with default parameters
    #[cfg_attr(feature = "wasm", wasm_bindgen(constructor))]
    pub fn new() -> Self {
        let mut parameters = std::collections::HashMap::new();
        let mut gas_costs = std::collections::HashMap::new();

        parameters.insert(0, PrivacyParameters {
            lattice_n: 0, modulus_q: 0, sigma: 0.0, security_bits: 0,
            proof_strategy: "none".to_string(), anonymity_set_size: None, decoy_count: None,
        });
        parameters.insert(1, PrivacyParameters {
            lattice_n: 256, modulus_q: 12289, sigma: 3.2, security_bits: 80,
            proof_strategy: "bulletproofs".to_string(), anonymity_set_size: None, decoy_count: Some(3),
        });
        parameters.insert(2, PrivacyParameters {
            lattice_n: 512, modulus_q: 12289, sigma: 3.2, security_bits: 128,
            proof_strategy: "bulletproofs".to_string(), anonymity_set_size: None, decoy_count: Some(7),
        });
        parameters.insert(3, PrivacyParameters {
            lattice_n: 1024, modulus_q: 40961, sigma: 3.2, security_bits: 192,
            proof_strategy: "quantum_lattice_pkc".to_string(), anonymity_set_size: Some(16), decoy_count: Some(15),
        });
        parameters.insert(4, PrivacyParameters {
            lattice_n: 2048, modulus_q: 65537, sigma: 3.2, security_bits: 256,
            proof_strategy: "quantum_lattice_pkc".to_string(), anonymity_set_size: Some(64), decoy_count: Some(31),
        });
        parameters.insert(5, PrivacyParameters {
            lattice_n: 4096, modulus_q: 786433, sigma: 3.2, security_bits: 256,
            proof_strategy: "hybrid_zksnark_lattice".to_string(), anonymity_set_size: Some(256), decoy_count: Some(63),
        });

        gas_costs.insert(0, 21000);
        gas_costs.insert(1, 50000);
        gas_costs.insert(2, 100000);
        gas_costs.insert(3, 200000);
        gas_costs.insert(4, 350000);
        gas_costs.insert(5, 500000);

        Self { parameters, gas_costs }
    }

    /// Get parameters for a specific privacy level
    pub fn get_parameters(&self, level: u8) -> Result<PrivacyParameters> {
        self.parameters
            .get(&level)
            .cloned()
            .ok_or(NexusZeroError::InvalidPrivacyLevel(level))
    }
}

impl PrivacyEngine {
    /// Recommend optimal privacy level based on transaction context
    pub fn recommend(&self, context: &TransactionContext) -> Result<PrivacyRecommendation> {
        let mut recommended_level: u8 = 3; // Default to Private
        let mut reasons: Vec<String> = Vec::new();
        let mut compliance_cap: Option<u8> = None;

        // Regulatory considerations - track as cap
        if context.requires_compliance {
            compliance_cap = Some(3);
            reasons.push("Regulatory compliance limits maximum privacy to Level 3".to_string());
        }

        // Transaction value considerations
        if context.value_usd > 100_000.0 {
            recommended_level = 5;
            reasons.push("Very high-value transaction ($100k+) warrants Sovereign privacy".to_string());
        } else if context.value_usd > 10_000.0 {
            recommended_level = recommended_level.max(4);
            reasons.push("High-value transaction ($10k+) benefits from Anonymous privacy".to_string());
        }

        // User preference
        if let Some(pref) = context.preferred_level {
            if pref <= 5 {
                recommended_level = pref;
                reasons.push(format!("User preference: Level {}", pref));
            }
        }

        // Risk score adjustment
        if context.risk_score > 0.7 {
            recommended_level = recommended_level.min(2);
            reasons.push("Elevated risk score (>0.7) reduces maximum privacy".to_string());
        }

        // Enforce compliance cap
        if let Some(cap) = compliance_cap {
            if recommended_level > cap {
                recommended_level = cap;
            }
        }

        let level_enum = PrivacyLevel::from_u8(recommended_level)?;

        Ok(PrivacyRecommendation {
            level: recommended_level,
            reasons,
            estimated_proof_time_ms: level_enum.estimated_proof_time_ms(),
            estimated_cost_gas: *self.gas_costs.get(&recommended_level).unwrap_or(&0),
        })
    }

    /// Check if morphing between privacy levels is possible
    pub fn can_morph(&self, from_level: u8, to_level: u8) -> Result<(bool, String)> {
        if from_level > 5 {
            return Err(NexusZeroError::InvalidPrivacyLevel(from_level));
        }
        if to_level > 5 {
            return Err(NexusZeroError::InvalidPrivacyLevel(to_level));
        }

        if to_level >= from_level {
            Ok((true, "Increasing or maintaining privacy level is always allowed".to_string()))
        } else {
            Ok((true, "Decreasing privacy must be done incrementally".to_string()))
        }
    }
}

impl Default for PrivacyEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_privacy_level_ordering() {
        assert!(PrivacyLevel::Transparent < PrivacyLevel::Sovereign);
        assert!(PrivacyLevel::Private < PrivacyLevel::Anonymous);
    }

    #[test]
    fn test_recommend_default() {
        let engine = PrivacyEngine::new();
        let context = TransactionContext::default();
        let rec = engine.recommend(&context).unwrap();
        assert_eq!(rec.level, 3); // Default is Private
    }

    #[test]
    fn test_recommend_compliance() {
        let engine = PrivacyEngine::new();
        let context = TransactionContext {
            value_usd: 50_000.0,
            requires_compliance: true,
            ..Default::default()
        };
        let rec = engine.recommend(&context).unwrap();
        assert!(rec.level <= 3); // Compliance caps at Private
    }
}
