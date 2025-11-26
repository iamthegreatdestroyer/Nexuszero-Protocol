//! SAR Processor Service

use crate::models::*;
use chrono::Utc;
use uuid::Uuid;

/// Service for SAR processing
pub struct SarProcessor;

impl Default for SarProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl SarProcessor {
    pub fn new() -> Self {
        Self
    }

    /// Create SAR from request
    pub fn create_sar(&self, request: &CreateSarRequest, created_by: Uuid) -> SuspiciousActivityReport {
        SuspiciousActivityReport {
            id: Uuid::new_v4(),
            reference_number: self.generate_reference(&request.jurisdiction),
            status: SarStatus::Draft,
            subject_id: request.subject_id,
            subject_type: request.subject_type.clone(),
            activity_type: request.activity_type.clone(),
            amount_involved: request.amount_involved,
            currency: request.currency.clone(),
            jurisdiction: request.jurisdiction.clone(),
            description: request.description.clone(),
            transaction_ids: request.transaction_ids.clone(),
            evidence: vec![],
            created_at: Utc::now(),
            updated_at: Utc::now(),
            submitted_at: None,
            created_by,
            reviewed_by: None,
        }
    }

    /// Generate SAR reference number
    fn generate_reference(&self, jurisdiction: &str) -> String {
        let year = Utc::now().format("%Y");
        let random = &Uuid::new_v4().to_string()[..8];
        format!("SAR-{}-{}-{}", jurisdiction, year, random.to_uppercase())
    }

    /// Validate SAR is complete for submission
    pub fn validate_for_submission(&self, sar: &SuspiciousActivityReport) -> Vec<String> {
        let mut errors = Vec::new();
        
        if sar.description.len() < 50 {
            errors.push("Description must be at least 50 characters".to_string());
        }
        if sar.transaction_ids.is_empty() {
            errors.push("At least one transaction must be linked".to_string());
        }
        if sar.evidence.is_empty() {
            errors.push("At least one evidence item must be attached".to_string());
        }
        
        errors
    }
}
