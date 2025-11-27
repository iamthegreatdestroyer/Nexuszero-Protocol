//! Compliance Integration for Privacy Morphing

use async_trait::async_trait;
use std::collections::HashMap;

use crate::config::ComplianceConfig;
use crate::error::{MorphingError, MorphingResult};
use crate::types::{PrivacyLevel, RecipientType};

/// Compliance integration for privacy ceiling enforcement
pub struct ComplianceIntegration {
    config: ComplianceConfig,
    /// Cache of entity compliance status
    entity_cache: HashMap<String, ComplianceStatus>,
}

/// Compliance status for an entity
#[derive(Debug, Clone)]
pub struct ComplianceStatus {
    /// Whether the entity is regulated
    pub is_regulated: bool,
    /// Applicable jurisdictions
    pub jurisdictions: Vec<String>,
    /// Maximum allowed privacy level
    pub max_privacy_level: PrivacyLevel,
    /// Whether attestation is required
    pub attestation_required: bool,
}

impl ComplianceIntegration {
    /// Create a new compliance integration
    pub fn new(config: ComplianceConfig) -> Self {
        Self {
            config,
            entity_cache: HashMap::new(),
        }
    }

    /// Get the privacy ceiling for a recipient type
    pub async fn get_privacy_ceiling(
        &self,
        recipient_type: &RecipientType,
    ) -> MorphingResult<PrivacyLevel> {
        if !self.config.enabled {
            return Ok(PrivacyLevel::new(10));
        }

        let ceiling = match recipient_type {
            RecipientType::Normal => PrivacyLevel::new(10),
            RecipientType::Exchange => self.config.regulated_max_level,
            RecipientType::Institutional => self.config.regulated_max_level,
            RecipientType::Contract => PrivacyLevel::new(8),
            RecipientType::Sanctioned => PrivacyLevel::new(1), // Force transparent
        };

        Ok(ceiling)
    }

    /// Check if an entity needs compliance
    pub async fn check_entity_compliance(
        &self,
        entity_id: &str,
    ) -> MorphingResult<ComplianceStatus> {
        // Check cache first
        if let Some(status) = self.entity_cache.get(entity_id) {
            return Ok(status.clone());
        }

        // In production, this would query a compliance database
        // For now, return default unregulated status
        Ok(ComplianceStatus {
            is_regulated: false,
            jurisdictions: Vec::new(),
            max_privacy_level: PrivacyLevel::new(10),
            attestation_required: false,
        })
    }

    /// Get privacy ceiling for a jurisdiction
    pub fn get_jurisdiction_ceiling(&self, jurisdiction: &str) -> PrivacyLevel {
        self.config.jurisdiction_overrides
            .get(jurisdiction)
            .copied()
            .unwrap_or(self.config.regulated_max_level)
    }

    /// Check if transaction requires compliance attestation
    pub async fn requires_attestation(
        &self,
        sender: &str,
        recipient: &str,
        amount: u128,
    ) -> MorphingResult<bool> {
        if !self.config.enabled {
            return Ok(false);
        }

        // Large transactions may require attestation
        let large_transaction_threshold = 10000 * 10u128.pow(18); // 10,000 tokens
        if amount >= large_transaction_threshold {
            return Ok(true);
        }

        // Check if either party is regulated
        let sender_status = self.check_entity_compliance(sender).await?;
        let recipient_status = self.check_entity_compliance(recipient).await?;

        Ok(sender_status.attestation_required || recipient_status.attestation_required)
    }

    /// Validate privacy level against compliance requirements
    pub async fn validate_privacy_level(
        &self,
        entity_id: &str,
        requested_level: PrivacyLevel,
    ) -> MorphingResult<PrivacyLevel> {
        let status = self.check_entity_compliance(entity_id).await?;

        if requested_level > status.max_privacy_level {
            Ok(status.max_privacy_level)
        } else {
            Ok(requested_level)
        }
    }
}

/// Trait for custom compliance providers
#[async_trait]
pub trait ComplianceProvider: Send + Sync {
    /// Check if an entity is sanctioned
    async fn is_sanctioned(&self, entity_id: &str) -> MorphingResult<bool>;

    /// Get compliance status for an entity
    async fn get_status(&self, entity_id: &str) -> MorphingResult<ComplianceStatus>;

    /// Verify compliance attestation
    async fn verify_attestation(&self, attestation_id: &str) -> MorphingResult<bool>;
}

/// Default compliance provider (no-op)
pub struct DefaultComplianceProvider;

#[async_trait]
impl ComplianceProvider for DefaultComplianceProvider {
    async fn is_sanctioned(&self, _entity_id: &str) -> MorphingResult<bool> {
        Ok(false)
    }

    async fn get_status(&self, _entity_id: &str) -> MorphingResult<ComplianceStatus> {
        Ok(ComplianceStatus {
            is_regulated: false,
            jurisdictions: Vec::new(),
            max_privacy_level: PrivacyLevel::new(10),
            attestation_required: false,
        })
    }

    async fn verify_attestation(&self, _attestation_id: &str) -> MorphingResult<bool> {
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_privacy_ceiling_by_recipient() {
        let config = ComplianceConfig::default();
        let compliance = ComplianceIntegration::new(config);

        let normal = compliance.get_privacy_ceiling(&RecipientType::Normal).await.unwrap();
        assert_eq!(normal.value(), 10);

        let exchange = compliance.get_privacy_ceiling(&RecipientType::Exchange).await.unwrap();
        assert!(exchange.value() < 10);

        let sanctioned = compliance.get_privacy_ceiling(&RecipientType::Sanctioned).await.unwrap();
        assert_eq!(sanctioned.value(), 1);
    }

    #[tokio::test]
    async fn test_entity_compliance() {
        let config = ComplianceConfig::default();
        let compliance = ComplianceIntegration::new(config);

        let status = compliance.check_entity_compliance("random_entity").await.unwrap();
        assert!(!status.is_regulated);
    }

    #[tokio::test]
    async fn test_attestation_requirements() {
        let config = ComplianceConfig::default();
        let compliance = ComplianceIntegration::new(config);

        // Small transaction
        let small = compliance.requires_attestation(
            "sender",
            "recipient",
            1000 * 10u128.pow(18),
        ).await.unwrap();
        assert!(!small);

        // Large transaction
        let large = compliance.requires_attestation(
            "sender",
            "recipient",
            100000 * 10u128.pow(18),
        ).await.unwrap();
        assert!(large);
    }
}
