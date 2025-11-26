//! Privacy Service data models

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Privacy level enum
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum PrivacyLevel {
    /// Fully transparent - no privacy
    Transparent = 0,
    /// Minimal privacy - basic obfuscation
    Minimal = 1,
    /// Standard privacy - moderate protection
    Standard = 2,
    /// Enhanced privacy - strong protection
    Enhanced = 3,
    /// Maximum privacy - full ZK protection
    Maximum = 4,
}

impl Default for PrivacyLevel {
    fn default() -> Self {
        Self::Standard
    }
}

impl PrivacyLevel {
    /// Get numeric value
    pub fn value(&self) -> u8 {
        *self as u8
    }

    /// Create from numeric value
    pub fn from_value(v: u8) -> Self {
        match v {
            0 => Self::Transparent,
            1 => Self::Minimal,
            2 => Self::Standard,
            3 => Self::Enhanced,
            4 => Self::Maximum,
            _ => Self::Standard,
        }
    }

    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Transparent => "Transparent",
            Self::Minimal => "Minimal",
            Self::Standard => "Standard",
            Self::Enhanced => "Enhanced",
            Self::Maximum => "Maximum",
        }
    }

    /// Get description
    pub fn description(&self) -> &'static str {
        match self {
            Self::Transparent => "No privacy protection, fully visible",
            Self::Minimal => "Basic obfuscation, addresses visible",
            Self::Standard => "Standard privacy with partial shielding",
            Self::Enhanced => "Enhanced privacy with strong protection",
            Self::Maximum => "Maximum privacy with full ZK protection",
        }
    }

    /// Get recommended proof type
    pub fn proof_type(&self) -> ProofType {
        match self {
            Self::Transparent => ProofType::None,
            Self::Minimal => ProofType::PartialZk,
            Self::Standard => ProofType::RangeProof,
            Self::Enhanced => ProofType::Groth16,
            Self::Maximum => ProofType::Groth16Plus,
        }
    }
}

/// Privacy level information (detailed)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyLevelInfo {
    /// Level enum value
    pub level: PrivacyLevel,

    /// Human-readable name
    pub name: String,

    /// Description
    pub description: String,

    /// Fields shielded at this level
    pub shielded_fields: Vec<String>,

    /// Proof type required
    pub proof_type: ProofType,

    /// Estimated gas cost
    pub estimated_gas: u64,

    /// Relative security score (0-100)
    pub security_score: u8,
}

/// Proof types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProofType {
    /// No proof required
    None,
    /// Partial ZK proof (single field)
    PartialZk,
    /// Range proof for amount
    RangeProof,
    /// Full Groth16 ZK-SNARK
    Groth16,
    /// Enhanced Groth16 with additional constraints
    Groth16Plus,
    /// Plonk proof system
    Plonk,
    /// Bulletproofs for range
    Bulletproofs,
    /// STARK proof
    Stark,
    /// Custom proof type
    Custom(String),
}

/// Privacy recommendation request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendationRequest {
    /// Sender address
    pub sender: String,

    /// Recipient address
    pub recipient: String,

    /// Transaction amount
    pub amount: i64,

    /// Chain identifier
    pub chain_id: String,

    /// Asset identifier
    pub asset_id: Option<String>,

    /// Transaction type hint
    pub transaction_type: Option<String>,

    /// User's privacy preference (1-10)
    pub privacy_preference: Option<u8>,
}

/// Privacy recommendation response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendationResponse {
    /// Recommended privacy level
    pub recommended_level: i16,

    /// Confidence score (0-1)
    pub confidence: f64,

    /// Factors that influenced the recommendation
    pub factors: Vec<RecommendationFactor>,

    /// Alternative recommendations
    pub alternatives: Vec<AlternativeRecommendation>,

    /// Cost comparison
    pub cost_analysis: CostAnalysis,
}

/// Recommendation factor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendationFactor {
    /// Factor name
    pub name: String,

    /// Impact level (low, medium, high)
    pub impact: String,

    /// Weight in decision (0-1)
    pub weight: f64,

    /// Reasoning
    pub reason: String,
}

/// Alternative recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlternativeRecommendation {
    /// Privacy level
    pub level: i16,

    /// Why this is an alternative
    pub reason: String,

    /// Score (0-1)
    pub score: f64,
}

/// Cost analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostAnalysis {
    /// Cost by level
    pub by_level: Vec<LevelCost>,

    /// Optimal balance level (cost vs privacy)
    pub optimal_balance: i16,
}

/// Cost for a privacy level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LevelCost {
    pub level: i16,
    pub gas: u64,
    pub fee_usd: f64,
    pub proof_time_ms: u64,
}

/// Privacy validation request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRequest {
    /// Requested privacy level
    pub privacy_level: i16,

    /// Transaction amount
    pub amount: i64,

    /// Chain identifier
    pub chain_id: String,

    /// Asset identifier
    pub asset_id: Option<String>,
}

/// Privacy validation response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResponse {
    /// Is the level valid
    pub valid: bool,

    /// Validation errors if any
    pub errors: Vec<String>,

    /// Warnings
    pub warnings: Vec<String>,

    /// Suggested level if current is invalid
    pub suggested_level: Option<i16>,
}

/// Privacy morph request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MorphRequest {
    /// Transaction ID
    pub transaction_id: Uuid,

    /// Current privacy level
    pub current_level: i16,

    /// Target privacy level
    pub target_level: i16,

    /// Current proof (if exists)
    pub current_proof: Option<String>,

    /// Force downgrade
    #[serde(default)]
    pub force: bool,

    /// Reason for morph
    pub reason: Option<String>,
}

/// Privacy morph response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MorphResponse {
    /// Morph operation ID
    pub morph_id: Uuid,

    /// Transaction ID
    pub transaction_id: Uuid,

    /// Previous level
    pub previous_level: i16,

    /// New level
    pub new_level: i16,

    /// New proof if generated
    pub new_proof: Option<String>,

    /// Whether new proof is required
    pub requires_new_proof: bool,

    /// Morph timestamp
    pub morphed_at: DateTime<Utc>,
}

/// Morph estimate request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MorphEstimateRequest {
    /// Current privacy level
    pub from_level: i16,

    /// Target privacy level
    pub to_level: i16,

    /// Current proof exists
    pub has_proof: bool,
}

/// Morph estimate response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MorphEstimateResponse {
    /// Can morph
    pub can_morph: bool,

    /// Requires new proof
    pub requires_new_proof: bool,

    /// Estimated gas cost
    pub estimated_gas: u64,

    /// Estimated fee in USD
    pub estimated_fee_usd: f64,

    /// Estimated time in ms
    pub estimated_time_ms: u64,

    /// Warnings
    pub warnings: Vec<String>,
}

/// Proof generation request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofGenerationRequest {
    /// Transaction ID
    pub transaction_id: Uuid,

    /// Sender address
    pub sender: String,

    /// Recipient address
    pub recipient: String,

    /// Transaction amount
    pub amount: i64,

    /// Privacy level
    pub privacy_level: i16,

    /// Priority
    #[serde(default)]
    pub priority: ProofPriority,

    /// Callback URL for async notification
    pub callback_url: Option<String>,
}

/// Proof priority
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProofPriority {
    Low,
    #[default]
    Normal,
    High,
}

/// Proof generation response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofGenerationResponse {
    /// Proof ID
    pub proof_id: Uuid,

    /// Transaction ID
    pub transaction_id: Uuid,

    /// Status
    pub status: ProofStatus,

    /// Proof data (if completed)
    pub proof: Option<String>,

    /// Verification key
    pub verification_key: Option<String>,

    /// Public inputs
    pub public_inputs: Option<Vec<String>>,

    /// Generation time in ms
    pub generation_time_ms: Option<u64>,

    /// Error if failed
    pub error: Option<String>,

    /// Created timestamp
    pub created_at: DateTime<Utc>,

    /// Completed timestamp
    pub completed_at: Option<DateTime<Utc>>,

    /// Queue position (if queued)
    pub queue_position: Option<u32>,

    /// Estimated wait time ms
    pub estimated_wait_ms: Option<u64>,
}

/// Proof status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProofStatus {
    Queued,
    Generating,
    Completed,
    Failed,
    Cancelled,
}

/// Proof verification request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofVerificationRequest {
    /// Proof data
    pub proof: String,

    /// Verification key
    pub verification_key: String,

    /// Public inputs
    pub public_inputs: Vec<String>,

    /// Privacy level (for circuit selection)
    pub privacy_level: i16,
}

/// Proof verification response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofVerificationResponse {
    /// Is valid
    pub valid: bool,

    /// Verification time ms
    pub verification_time_ms: u64,

    /// Error if invalid
    pub error: Option<String>,
}

/// Selective disclosure request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisclosureRequest {
    /// Transaction ID
    pub transaction_id: Uuid,

    /// Privacy level of transaction
    pub privacy_level: i16,

    /// Original proof
    pub proof: Option<String>,

    /// Fields to disclose
    pub fields: Vec<String>,

    /// Recipient of disclosure
    pub recipient_id: String,

    /// Purpose
    pub purpose: String,

    /// Expiry
    pub expires_at: Option<DateTime<Utc>>,
}

/// Selective disclosure response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisclosureResponse {
    /// Disclosure ID
    pub disclosure_id: Uuid,

    /// Transaction ID
    pub transaction_id: Uuid,

    /// Disclosed fields
    pub fields: Vec<String>,

    /// Disclosure proof
    pub proof: String,

    /// Recipient
    pub recipient_id: String,

    /// Purpose
    pub purpose: String,

    /// Valid until
    pub expires_at: Option<DateTime<Utc>>,

    /// Created at
    pub created_at: DateTime<Utc>,
}

/// Batch proof request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchProofRequest {
    /// Proof requests
    pub proofs: Vec<ProofGenerationRequest>,

    /// Fail on first error
    #[serde(default)]
    pub fail_fast: bool,
}

/// Batch proof response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchProofResponse {
    /// Batch ID
    pub batch_id: Uuid,

    /// Total proofs
    pub total: usize,

    /// Queued count
    pub queued: usize,

    /// Failed count
    pub failed: usize,

    /// Individual results
    pub results: Vec<BatchProofResult>,
}

/// Individual batch result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchProofResult {
    pub index: usize,
    pub proof_id: Option<Uuid>,
    pub status: ProofStatus,
    pub error: Option<String>,
}

/// Get all defined privacy levels
pub fn get_all_privacy_levels() -> Vec<PrivacyLevelInfo> {
    vec![
        PrivacyLevelInfo {
            level: PrivacyLevel::Transparent,
            name: "Transparent".to_string(),
            description: "All transaction data is publicly visible on the blockchain".to_string(),
            shielded_fields: vec![],
            proof_type: ProofType::None,
            estimated_gas: 21000,
            security_score: 0,
        },
        PrivacyLevelInfo {
            level: PrivacyLevel::Minimal,
            name: "Minimal".to_string(),
            description: "Sender address is hidden, recipient and amount are visible".to_string(),
            shielded_fields: vec!["sender".to_string()],
            proof_type: ProofType::PartialZk,
            estimated_gas: 71000,
            security_score: 25,
        },
        PrivacyLevelInfo {
            level: PrivacyLevel::Standard,
            name: "Standard".to_string(),
            description: "Recipient address is hidden, sender and amount are visible".to_string(),
            shielded_fields: vec!["recipient".to_string()],
            proof_type: ProofType::PartialZk,
            estimated_gas: 71000,
            security_score: 25,
        },
        PrivacyLevelInfo {
            level: PrivacyLevel::Enhanced,
            name: "Enhanced".to_string(),
            description: "Transaction amount is hidden, parties are visible".to_string(),
            shielded_fields: vec!["amount".to_string()],
            proof_type: ProofType::RangeProof,
            estimated_gas: 96000,
            security_score: 40,
        },
        PrivacyLevelInfo {
            level: PrivacyLevel::Maximum,
            name: "Maximum Privacy".to_string(),
            description: "Sender, recipient, and amount are all hidden."
                .to_string(),
            shielded_fields: vec![
                "sender".to_string(),
                "recipient".to_string(),
                "amount".to_string(),
                "memo".to_string(),
            ],
            proof_type: ProofType::Groth16Plus,
            estimated_gas: 271000,
            security_score: 100,
        },
    ]
}

/// Type aliases for backward compatibility
pub type ProofRequest = ProofGenerationRequest;
pub type ProofResponse = ProofGenerationResponse;
pub type VerificationResult = ProofVerificationResponse;
pub type MorphEstimate = MorphEstimateResponse;
pub type MorphStatus = ProofStatus;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_privacy_levels() {
        let levels = get_all_privacy_levels();
        assert_eq!(levels.len(), 5);
    }
}
