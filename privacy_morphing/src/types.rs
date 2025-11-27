//! Privacy Morphing Types

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Privacy level (1-10 scale)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PrivacyLevel(u8);

impl PrivacyLevel {
    /// Create a new privacy level (clamped to 1-10)
    pub fn new(level: u8) -> Self {
        Self(level.clamp(1, 10))
    }

    /// Get the raw value
    pub fn value(&self) -> u8 {
        self.0
    }

    /// Check if this is maximum privacy
    pub fn is_max(&self) -> bool {
        self.0 == 10
    }

    /// Check if this is minimum privacy (transparent)
    pub fn is_min(&self) -> bool {
        self.0 == 1
    }

    /// Get the corresponding anonymity set size requirement
    pub fn required_anonymity_set_size(&self) -> usize {
        match self.0 {
            1 => 1,      // Transparent
            2 => 5,
            3 => 10,
            4 => 25,
            5 => 50,
            6 => 100,
            7 => 250,
            8 => 500,
            9 => 1000,
            10 => 2500,  // Maximum privacy
            _ => 50,     // Default
        }
    }

    /// Get ring size for this privacy level
    pub fn ring_size(&self) -> usize {
        match self.0 {
            1 => 1,
            2..=3 => 4,
            4..=5 => 8,
            6..=7 => 16,
            8..=9 => 32,
            10 => 64,
            _ => 8,
        }
    }
}

impl Default for PrivacyLevel {
    fn default() -> Self {
        Self(5)
    }
}

impl From<u8> for PrivacyLevel {
    fn from(v: u8) -> Self {
        Self::new(v)
    }
}

/// Transaction context for privacy calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionContext {
    /// Transaction amount in base units
    pub amount: u128,
    /// Recipient type
    pub recipient_type: RecipientType,
    /// Whether compliance attestation is required
    pub compliance_required: bool,
    /// Transaction urgency (affects mixing time)
    pub urgency: TransactionUrgency,
    /// Source chain
    pub source_chain: Option<String>,
    /// Destination chain (for bridge transactions)
    pub dest_chain: Option<String>,
    /// User-requested privacy level
    pub requested_level: Option<PrivacyLevel>,
}

impl Default for TransactionContext {
    fn default() -> Self {
        Self {
            amount: 0,
            recipient_type: RecipientType::Normal,
            compliance_required: false,
            urgency: TransactionUrgency::Normal,
            source_chain: None,
            dest_chain: None,
            requested_level: None,
        }
    }
}

/// Recipient type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecipientType {
    /// Normal user address
    Normal,
    /// Exchange address (known)
    Exchange,
    /// Contract address
    Contract,
    /// Institutional address
    Institutional,
    /// Sanctioned (blocked)
    Sanctioned,
}

/// Transaction urgency
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransactionUrgency {
    /// Allow full mixing time
    Low,
    /// Standard mixing time
    Normal,
    /// Reduced mixing time
    High,
    /// Immediate (minimal mixing)
    Immediate,
}

impl TransactionUrgency {
    /// Get mixing time in seconds
    pub fn mixing_time_seconds(&self) -> u64 {
        match self {
            TransactionUrgency::Low => 3600,       // 1 hour
            TransactionUrgency::Normal => 600,     // 10 minutes
            TransactionUrgency::High => 60,        // 1 minute
            TransactionUrgency::Immediate => 0,    // No delay
        }
    }
}

/// Privacy morphing result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyCalculationResult {
    /// Calculated privacy level
    pub privacy_level: PrivacyLevel,
    /// Anonymity set to use
    pub anonymity_set_id: Uuid,
    /// Anonymity set size
    pub anonymity_set_size: usize,
    /// Ring size for transaction
    pub ring_size: usize,
    /// Recommended mixing delay
    pub mixing_delay_seconds: u64,
    /// Whether a compliance proof is attached
    pub compliance_attached: bool,
    /// Maximum privacy ceiling (if limited)
    pub privacy_ceiling: Option<PrivacyLevel>,
    /// Calculation timestamp
    pub calculated_at: DateTime<Utc>,
}

/// Privacy adjustment event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyAdjustment {
    /// Unique adjustment ID
    pub id: Uuid,
    /// Previous privacy level
    pub from_level: PrivacyLevel,
    /// New privacy level
    pub to_level: PrivacyLevel,
    /// Reason for adjustment
    pub reason: AdjustmentReason,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Reasons for privacy level adjustment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdjustmentReason {
    /// Scheduled morphing
    ScheduledMorph { schedule_id: Uuid },
    /// User request
    UserRequest,
    /// Compliance requirement
    ComplianceRequirement { jurisdiction: String },
    /// Network congestion
    NetworkCongestion { level: f64 },
    /// Anonymity set availability
    AnonymitySetChange { new_size: usize },
    /// Security policy
    SecurityPolicy { policy_name: String },
}

/// Privacy profile for a user/account
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyProfile {
    /// Profile ID
    pub id: Uuid,
    /// Account identifier
    pub account_id: String,
    /// Default privacy level
    pub default_level: PrivacyLevel,
    /// Minimum allowed privacy level
    pub min_level: PrivacyLevel,
    /// Maximum allowed privacy level
    pub max_level: PrivacyLevel,
    /// Active morphing schedule
    pub active_schedule: Option<Uuid>,
    /// Compliance requirements
    pub compliance_jurisdictions: Vec<String>,
    /// Created at
    pub created_at: DateTime<Utc>,
    /// Last updated
    pub updated_at: DateTime<Utc>,
}

impl PrivacyProfile {
    /// Create a new profile with defaults
    pub fn new(account_id: String) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            account_id,
            default_level: PrivacyLevel::default(),
            min_level: PrivacyLevel::new(1),
            max_level: PrivacyLevel::new(10),
            active_schedule: None,
            compliance_jurisdictions: Vec::new(),
            created_at: now,
            updated_at: now,
        }
    }

    /// Check if a privacy level is allowed for this profile
    pub fn is_level_allowed(&self, level: PrivacyLevel) -> bool {
        level >= self.min_level && level <= self.max_level
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_privacy_level_clamping() {
        assert_eq!(PrivacyLevel::new(0).value(), 1);
        assert_eq!(PrivacyLevel::new(5).value(), 5);
        assert_eq!(PrivacyLevel::new(15).value(), 10);
    }

    #[test]
    fn test_privacy_level_ring_size() {
        assert_eq!(PrivacyLevel::new(1).ring_size(), 1);
        assert_eq!(PrivacyLevel::new(5).ring_size(), 8);
        assert_eq!(PrivacyLevel::new(10).ring_size(), 64);
    }

    #[test]
    fn test_privacy_profile_level_check() {
        let mut profile = PrivacyProfile::new("test".to_string());
        profile.min_level = PrivacyLevel::new(3);
        profile.max_level = PrivacyLevel::new(8);

        assert!(!profile.is_level_allowed(PrivacyLevel::new(1)));
        assert!(profile.is_level_allowed(PrivacyLevel::new(5)));
        assert!(!profile.is_level_allowed(PrivacyLevel::new(10)));
    }

    #[test]
    fn test_transaction_urgency_mixing_time() {
        assert!(TransactionUrgency::Low.mixing_time_seconds() > 
                TransactionUrgency::Normal.mixing_time_seconds());
        assert_eq!(TransactionUrgency::Immediate.mixing_time_seconds(), 0);
    }
}
