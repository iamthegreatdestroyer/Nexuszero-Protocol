//! Transaction data models
//!
//! Core data structures for the NexusZero transaction system

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::FromRow;
use uuid::Uuid;

/// Privacy levels for transactions (0-5)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, sqlx::Type)]
#[repr(i16)]
pub enum PrivacyLevel {
    /// Level 0: Full transparency - all data public
    #[serde(rename = "transparent")]
    Transparent = 0,

    /// Level 1: Sender-shielded - recipient and amount visible
    #[serde(rename = "sender_shielded")]
    SenderShielded = 1,

    /// Level 2: Recipient-shielded - sender and amount visible
    #[serde(rename = "recipient_shielded")]
    RecipientShielded = 2,

    /// Level 3: Amount-shielded - parties visible
    #[serde(rename = "amount_shielded")]
    AmountShielded = 3,

    /// Level 4: Full privacy - only memo visible
    #[serde(rename = "full_privacy")]
    FullPrivacy = 4,

    /// Level 5: Maximum privacy - all data shielded
    #[serde(rename = "maximum")]
    Maximum = 5,
}

impl Default for PrivacyLevel {
    fn default() -> Self {
        Self::FullPrivacy
    }
}

impl PrivacyLevel {
    /// Get the numeric value of the privacy level
    pub fn as_i16(&self) -> i16 {
        *self as i16
    }

    /// Create from numeric value
    pub fn from_i16(value: i16) -> Option<Self> {
        match value {
            0 => Some(Self::Transparent),
            1 => Some(Self::SenderShielded),
            2 => Some(Self::RecipientShielded),
            3 => Some(Self::AmountShielded),
            4 => Some(Self::FullPrivacy),
            5 => Some(Self::Maximum),
            _ => None,
        }
    }

    /// Check if sender is shielded at this level
    pub fn sender_shielded(&self) -> bool {
        matches!(
            self,
            Self::SenderShielded | Self::FullPrivacy | Self::Maximum
        )
    }

    /// Check if recipient is shielded at this level
    pub fn recipient_shielded(&self) -> bool {
        matches!(
            self,
            Self::RecipientShielded | Self::FullPrivacy | Self::Maximum
        )
    }

    /// Check if amount is shielded at this level
    pub fn amount_shielded(&self) -> bool {
        matches!(
            self,
            Self::AmountShielded | Self::FullPrivacy | Self::Maximum
        )
    }
}

/// Transaction status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "transaction_status", rename_all = "snake_case")]
pub enum TransactionStatus {
    /// Transaction created, awaiting proof
    Pending,

    /// Proof generation in progress
    ProofGenerating,

    /// Proof generated, awaiting submission
    ProofReady,

    /// Submitted to blockchain
    Submitted,

    /// Confirmed on blockchain
    Confirmed,

    /// Transaction finalized
    Finalized,

    /// Transaction failed
    Failed,

    /// Transaction cancelled
    Cancelled,
}

impl Default for TransactionStatus {
    fn default() -> Self {
        Self::Pending
    }
}

/// Core transaction record
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Transaction {
    /// Unique transaction ID
    pub id: Uuid,

    /// User/wallet ID that created the transaction
    pub user_id: Uuid,

    /// Sender address (may be encrypted based on privacy level)
    pub sender: String,

    /// Recipient address (may be encrypted based on privacy level)
    pub recipient: String,

    /// Transaction amount in base units
    pub amount: i64,

    /// Asset identifier
    pub asset_id: String,

    /// Privacy level (0-5)
    pub privacy_level: i16,

    /// Current status
    pub status: TransactionStatus,

    /// Chain identifier
    pub chain_id: String,

    /// On-chain transaction hash (if submitted)
    pub chain_tx_hash: Option<String>,

    /// ZK proof (base64 encoded)
    pub proof: Option<String>,

    /// Proof ID for tracking
    pub proof_id: Option<Uuid>,

    /// Encrypted memo
    pub memo: Option<String>,

    /// Metadata JSON
    pub metadata: Option<serde_json::Value>,

    /// Error message if failed
    pub error_message: Option<String>,

    /// Block number if confirmed
    pub block_number: Option<i64>,

    /// Gas used
    pub gas_used: Option<i64>,

    /// Created timestamp
    pub created_at: DateTime<Utc>,

    /// Last updated timestamp
    pub updated_at: DateTime<Utc>,

    /// Finalized timestamp
    pub finalized_at: Option<DateTime<Utc>>,
}

/// Request to create a new transaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateTransactionRequest {
    /// Sender address
    pub sender: String,

    /// Recipient address
    pub recipient: String,

    /// Amount in base units
    pub amount: i64,

    /// Asset identifier
    pub asset_id: String,

    /// Requested privacy level (0-5)
    #[serde(default)]
    pub privacy_level: Option<i16>,

    /// Chain identifier
    pub chain_id: String,

    /// Optional memo
    pub memo: Option<String>,

    /// Optional metadata
    pub metadata: Option<serde_json::Value>,

    /// Auto-generate proof after creation
    #[serde(default)]
    pub auto_generate_proof: bool,
}

/// Response after creating a transaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateTransactionResponse {
    /// Transaction ID
    pub id: Uuid,

    /// Current status
    pub status: TransactionStatus,

    /// Assigned privacy level
    pub privacy_level: i16,

    /// Proof ID if auto-generation started
    pub proof_id: Option<Uuid>,

    /// Created timestamp
    pub created_at: DateTime<Utc>,
}

/// Transaction update request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateTransactionRequest {
    /// New memo
    pub memo: Option<String>,

    /// New metadata
    pub metadata: Option<serde_json::Value>,
}

/// Transaction list query parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListTransactionsQuery {
    /// Filter by status
    pub status: Option<TransactionStatus>,

    /// Filter by privacy level
    pub privacy_level: Option<i16>,

    /// Filter by chain
    pub chain_id: Option<String>,

    /// Filter by asset
    pub asset_id: Option<String>,

    /// Filter after date
    pub created_after: Option<DateTime<Utc>>,

    /// Filter before date
    pub created_before: Option<DateTime<Utc>>,

    /// Page number (1-indexed)
    #[serde(default = "default_page")]
    pub page: u32,

    /// Page size
    #[serde(default = "default_page_size")]
    pub page_size: u32,

    /// Sort field
    #[serde(default = "default_sort_field")]
    pub sort_by: String,

    /// Sort direction
    #[serde(default)]
    pub sort_desc: bool,
}

fn default_page() -> u32 {
    1
}

fn default_page_size() -> u32 {
    20
}

fn default_sort_field() -> String {
    "created_at".to_string()
}

/// Paginated transaction list response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionListResponse {
    /// Transactions
    pub transactions: Vec<TransactionSummary>,

    /// Total count
    pub total: i64,

    /// Current page
    pub page: u32,

    /// Page size
    pub page_size: u32,

    /// Total pages
    pub total_pages: u32,
}

/// Transaction summary for list views
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionSummary {
    /// Transaction ID
    pub id: Uuid,

    /// Status
    pub status: TransactionStatus,

    /// Privacy level
    pub privacy_level: i16,

    /// Amount (may be hidden based on privacy)
    pub amount: Option<i64>,

    /// Asset
    pub asset_id: String,

    /// Chain
    pub chain_id: String,

    /// Has proof
    pub has_proof: bool,

    /// Created timestamp
    pub created_at: DateTime<Utc>,
}

/// Proof request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofRequest {
    /// Priority level
    #[serde(default)]
    pub priority: ProofPriority,

    /// Callback URL for async notification
    pub callback_url: Option<String>,
}

/// Proof priority levels
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProofPriority {
    /// Low priority (batch processing)
    Low,

    /// Normal priority
    #[default]
    Normal,

    /// High priority (immediate processing)
    High,
}

/// Proof response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofResponse {
    /// Proof ID
    pub proof_id: Uuid,

    /// Transaction ID
    pub transaction_id: Uuid,

    /// Status
    pub status: ProofStatus,

    /// Proof data (base64 encoded)
    pub proof: Option<String>,

    /// Verification key
    pub verification_key: Option<String>,

    /// Public inputs
    pub public_inputs: Option<Vec<String>>,

    /// Generation time in milliseconds
    pub generation_time_ms: Option<u64>,

    /// Error if failed
    pub error: Option<String>,

    /// Created timestamp
    pub created_at: DateTime<Utc>,

    /// Completed timestamp
    pub completed_at: Option<DateTime<Utc>>,
}

/// Proof status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProofStatus {
    /// Queued for generation
    Queued,

    /// Currently generating
    Generating,

    /// Successfully generated
    Completed,

    /// Generation failed
    Failed,

    /// Cancelled
    Cancelled,
}

/// Batch transaction request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchTransactionRequest {
    /// List of transactions to create
    pub transactions: Vec<CreateTransactionRequest>,

    /// Generate proofs for all
    #[serde(default)]
    pub generate_proofs: bool,

    /// Atomic - all or nothing
    #[serde(default)]
    pub atomic: bool,
}

/// Batch transaction response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchTransactionResponse {
    /// Batch ID
    pub batch_id: Uuid,

    /// Total transactions
    pub total: usize,

    /// Successfully created
    pub created: usize,

    /// Failed count
    pub failed: usize,

    /// Transaction results
    pub results: Vec<BatchTransactionResult>,
}

/// Individual batch result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchTransactionResult {
    /// Index in original request
    pub index: usize,

    /// Success status
    pub success: bool,

    /// Transaction ID if created
    pub transaction_id: Option<Uuid>,

    /// Error message if failed
    pub error: Option<String>,
}

/// Privacy morph request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyMorphRequest {
    /// Target privacy level
    pub target_level: i16,

    /// Reason for morph (for audit)
    pub reason: Option<String>,

    /// Force morph even if level lower
    #[serde(default)]
    pub force: bool,
}

/// Privacy morph response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyMorphResponse {
    /// Transaction ID
    pub transaction_id: Uuid,

    /// Previous privacy level
    pub previous_level: i16,

    /// New privacy level
    pub new_level: i16,

    /// New proof ID (if re-proof needed)
    pub proof_id: Option<Uuid>,

    /// Morphed timestamp
    pub morphed_at: DateTime<Utc>,
}

/// Selective disclosure request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectiveDisclosureRequest {
    /// Fields to disclose
    pub fields: Vec<String>,

    /// Recipient of disclosure
    pub recipient_id: String,

    /// Expiry timestamp
    pub expires_at: Option<DateTime<Utc>>,

    /// Purpose of disclosure
    pub purpose: String,
}

/// Selective disclosure response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectiveDisclosureResponse {
    /// Disclosure ID
    pub disclosure_id: Uuid,

    /// Transaction ID
    pub transaction_id: Uuid,

    /// Disclosed fields
    pub fields: Vec<String>,

    /// Disclosure proof
    pub proof: String,

    /// Valid until
    pub expires_at: Option<DateTime<Utc>>,

    /// Created timestamp
    pub created_at: DateTime<Utc>,
}

/// Compliance status for a transaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceStatus {
    /// Transaction ID
    pub transaction_id: Uuid,

    /// Is compliant
    pub compliant: bool,

    /// Compliance checks performed
    pub checks: Vec<ComplianceCheck>,

    /// Overall risk score (0-100)
    pub risk_score: u8,

    /// Regulatory flags
    pub flags: Vec<String>,

    /// Last checked
    pub checked_at: DateTime<Utc>,
}

/// Individual compliance check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceCheck {
    /// Check name
    pub name: String,

    /// Check passed
    pub passed: bool,

    /// Check details
    pub details: Option<String>,
}

/// Analytics summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsSummary {
    /// Total transactions
    pub total_transactions: i64,

    /// Transactions by status
    pub by_status: std::collections::HashMap<String, i64>,

    /// Transactions by privacy level
    pub by_privacy_level: std::collections::HashMap<i16, i64>,

    /// Total volume
    pub total_volume: i64,

    /// Average proof generation time (ms)
    pub avg_proof_time_ms: f64,

    /// Success rate (percentage)
    pub success_rate: f64,

    /// Time period start
    pub period_start: DateTime<Utc>,

    /// Time period end
    pub period_end: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_privacy_level_conversion() {
        assert_eq!(PrivacyLevel::Transparent.as_i16(), 0);
        assert_eq!(PrivacyLevel::Maximum.as_i16(), 5);

        assert_eq!(
            PrivacyLevel::from_i16(3),
            Some(PrivacyLevel::AmountShielded)
        );
        assert_eq!(PrivacyLevel::from_i16(99), None);
    }

    #[test]
    fn test_privacy_level_shielding() {
        assert!(!PrivacyLevel::Transparent.sender_shielded());
        assert!(PrivacyLevel::SenderShielded.sender_shielded());
        assert!(PrivacyLevel::FullPrivacy.sender_shielded());

        assert!(!PrivacyLevel::Transparent.amount_shielded());
        assert!(PrivacyLevel::AmountShielded.amount_shielded());
        assert!(PrivacyLevel::Maximum.amount_shielded());
    }

    #[test]
    fn test_default_privacy_level() {
        let level = PrivacyLevel::default();
        assert_eq!(level, PrivacyLevel::FullPrivacy);
    }

    #[test]
    fn test_transaction_status_default() {
        let status = TransactionStatus::default();
        assert_eq!(status, TransactionStatus::Pending);
    }
}
