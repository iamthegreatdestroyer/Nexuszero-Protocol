//! Data models for Compliance Service

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Compliance check status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "compliance_status", rename_all = "snake_case")]
#[serde(rename_all = "snake_case")]
pub enum ComplianceStatus {
    Pending,
    Approved,
    Rejected,
    RequiresReview,
    Escalated,
    Expired,
}

/// Risk level classification
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "risk_level", rename_all = "snake_case")]
#[serde(rename_all = "snake_case")]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
    Blocked,
}

impl RiskLevel {
    pub fn from_score(score: f64) -> Self {
        match score {
            s if s < 0.25 => RiskLevel::Low,
            s if s < 0.5 => RiskLevel::Medium,
            s if s < 0.75 => RiskLevel::High,
            s if s < 0.9 => RiskLevel::Critical,
            _ => RiskLevel::Blocked,
        }
    }
}

/// SAR (Suspicious Activity Report) status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "sar_status", rename_all = "snake_case")]
#[serde(rename_all = "snake_case")]
pub enum SarStatus {
    Draft,
    PendingReview,
    Approved,
    Submitted,
    Acknowledged,
    Rejected,
}

/// KYC status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "kyc_status", rename_all = "snake_case")]
#[serde(rename_all = "snake_case")]
pub enum KycStatus {
    NotStarted,
    InProgress,
    Pending,
    Verified,
    Failed,
    Expired,
}

/// Report type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "report_type", rename_all = "snake_case")]
#[serde(rename_all = "snake_case")]
pub enum ReportType {
    Daily,
    Weekly,
    Monthly,
    Quarterly,
    Annual,
    AdHoc,
    Regulatory,
}

/// Rule type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "rule_type", rename_all = "snake_case")]
#[serde(rename_all = "snake_case")]
pub enum RuleType {
    Threshold,
    Pattern,
    Velocity,
    Watchlist,
    Geographic,
    Behavioral,
    Custom,
}

/// Compliance check request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceCheckRequest {
    pub transaction_id: Uuid,
    pub sender_id: Uuid,
    pub recipient_id: Uuid,
    pub amount: u64,
    pub currency: String,
    pub transaction_type: String,
    pub jurisdiction: String,
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

/// Compliance check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceCheckResult {
    pub id: Uuid,
    pub transaction_id: Uuid,
    pub status: ComplianceStatus,
    pub risk_level: RiskLevel,
    pub risk_score: f64,
    pub rules_checked: Vec<String>,
    pub rules_triggered: Vec<RuleViolation>,
    pub recommendations: Vec<String>,
    pub requires_sar: bool,
    pub checked_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
    pub metadata: HashMap<String, String>,
}

/// Rule violation details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleViolation {
    pub rule_id: String,
    pub rule_name: String,
    pub severity: RiskLevel,
    pub description: String,
    pub jurisdiction: String,
    pub remediation: Option<String>,
}

/// Batch compliance check request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchComplianceRequest {
    pub transactions: Vec<ComplianceCheckRequest>,
    #[serde(default)]
    pub parallel: bool,
}

/// Batch compliance check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchComplianceResult {
    pub batch_id: Uuid,
    pub results: Vec<ComplianceCheckResult>,
    pub summary: BatchSummary,
}

/// Summary of batch compliance check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchSummary {
    pub total: usize,
    pub approved: usize,
    pub rejected: usize,
    pub requires_review: usize,
    pub processing_time_ms: u64,
}

/// Risk assessment request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessmentRequest {
    pub entity_id: Uuid,
    pub entity_type: String, // "individual", "business", "address"
    pub jurisdiction: String,
    #[serde(default)]
    pub include_history: bool,
    #[serde(default)]
    pub context: HashMap<String, String>,
}

/// Risk assessment result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessment {
    pub entity_id: Uuid,
    pub risk_score: f64,
    pub risk_level: RiskLevel,
    pub factors: Vec<RiskFactor>,
    pub recommendations: Vec<String>,
    pub assessed_at: DateTime<Utc>,
    pub valid_until: DateTime<Utc>,
}

/// Individual risk factor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskFactor {
    pub name: String,
    pub category: String,
    pub weight: f64,
    pub score: f64,
    pub description: String,
}

/// SAR (Suspicious Activity Report)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuspiciousActivityReport {
    pub id: Uuid,
    pub reference_number: String,
    pub status: SarStatus,
    pub subject_id: Uuid,
    pub subject_type: String,
    pub activity_type: String,
    pub amount_involved: u64,
    pub currency: String,
    pub jurisdiction: String,
    pub description: String,
    pub transaction_ids: Vec<Uuid>,
    pub evidence: Vec<SarEvidence>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub submitted_at: Option<DateTime<Utc>>,
    pub created_by: Uuid,
    pub reviewed_by: Option<Uuid>,
}

/// SAR evidence item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SarEvidence {
    pub evidence_type: String,
    pub description: String,
    pub reference: String,
    pub timestamp: DateTime<Utc>,
}

/// Create SAR request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateSarRequest {
    pub subject_id: Uuid,
    pub subject_type: String,
    pub activity_type: String,
    pub amount_involved: u64,
    pub currency: String,
    pub jurisdiction: String,
    pub description: String,
    pub transaction_ids: Vec<Uuid>,
}

/// KYC verification request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KycVerificationRequest {
    pub entity_id: Uuid,
    pub entity_type: String,
    pub full_name: String,
    pub date_of_birth: Option<String>,
    pub nationality: Option<String>,
    pub address: Option<String>,
    pub id_type: String,
    pub id_number: String,
    pub id_expiry: Option<String>,
    #[serde(default)]
    pub documents: Vec<String>, // Document references
}

/// KYC verification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KycVerificationResult {
    pub entity_id: Uuid,
    pub status: KycStatus,
    pub verification_level: u8, // 0-3
    pub checks_passed: Vec<String>,
    pub checks_failed: Vec<String>,
    pub requires_manual_review: bool,
    pub verified_at: Option<DateTime<Utc>>,
    pub expires_at: Option<DateTime<Utc>>,
}

/// AML screening request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmlScreeningRequest {
    pub entity_id: Uuid,
    pub name: String,
    pub entity_type: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    pub date_of_birth: Option<String>,
    pub nationality: Option<String>,
    pub jurisdiction: String,
}

/// AML screening result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmlScreeningResult {
    pub entity_id: Uuid,
    pub screened_at: DateTime<Utc>,
    pub matches: Vec<WatchlistMatch>,
    pub risk_indicators: Vec<String>,
    pub overall_status: String, // "clear", "match", "potential_match"
}

/// Watchlist match
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatchlistMatch {
    pub list_name: String,
    pub list_type: String, // "sanctions", "pep", "adverse_media"
    pub match_score: f64,
    pub matched_name: String,
    pub matched_fields: Vec<String>,
    pub source_url: Option<String>,
}

/// Travel Rule originator data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TravelRuleOriginator {
    pub transfer_id: Uuid,
    pub originator_vasp_id: String,
    pub originator_name: String,
    pub originator_account: String,
    pub originator_address: Option<String>,
    pub originator_location: String,
    pub originator_id_type: Option<String>,
    pub originator_id_number: Option<String>,
}

/// Travel Rule beneficiary data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TravelRuleBeneficiary {
    pub transfer_id: Uuid,
    pub beneficiary_vasp_id: String,
    pub beneficiary_name: String,
    pub beneficiary_account: String,
    pub beneficiary_address: Option<String>,
}

/// Travel Rule transfer verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TravelRuleVerification {
    pub transfer_id: Uuid,
    pub is_compliant: bool,
    pub missing_fields: Vec<String>,
    pub warnings: Vec<String>,
    pub verified_at: DateTime<Utc>,
}

/// VASP (Virtual Asset Service Provider) info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VaspInfo {
    pub vasp_id: String,
    pub name: String,
    pub jurisdiction: String,
    pub registration_number: Option<String>,
    pub is_registered: bool,
    pub travel_rule_capable: bool,
    pub protocols_supported: Vec<String>,
    pub endpoint_url: Option<String>,
}

/// Compliance rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceRule {
    pub id: Uuid,
    pub name: String,
    pub description: String,
    pub rule_type: RuleType,
    pub jurisdiction: String,
    pub severity: RiskLevel,
    pub conditions: serde_json::Value,
    pub actions: Vec<String>,
    pub is_active: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Create rule request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateRuleRequest {
    pub name: String,
    pub description: String,
    pub rule_type: RuleType,
    pub jurisdiction: String,
    pub severity: RiskLevel,
    pub conditions: serde_json::Value,
    pub actions: Vec<String>,
}

/// Rule test request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleTestRequest {
    pub test_data: serde_json::Value,
}

/// Rule test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleTestResult {
    pub rule_id: Uuid,
    pub triggered: bool,
    pub matched_conditions: Vec<String>,
    pub actions_to_execute: Vec<String>,
    pub execution_time_ms: u64,
}

/// Compliance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceReport {
    pub id: Uuid,
    pub report_type: ReportType,
    pub jurisdiction: String,
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
    pub generated_at: DateTime<Utc>,
    pub summary: ReportSummary,
    pub status: String,
    pub file_url: Option<String>,
}

/// Report summary statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSummary {
    pub total_transactions: u64,
    pub transactions_approved: u64,
    pub transactions_rejected: u64,
    pub sars_filed: u64,
    pub high_risk_entities: u64,
    pub watchlist_matches: u64,
    pub average_risk_score: f64,
}

/// Report generation request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateReportRequest {
    pub report_type: ReportType,
    pub jurisdiction: String,
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
    #[serde(default)]
    pub include_details: bool,
}

/// Report schedule request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduleReportRequest {
    pub report_type: ReportType,
    pub jurisdiction: String,
    pub cron_expression: String,
    pub recipients: Vec<String>,
}

/// Audit log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditLogEntry {
    pub id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub action: String,
    pub entity_type: String,
    pub entity_id: Uuid,
    pub actor_id: Option<Uuid>,
    pub actor_type: String,
    pub details: serde_json::Value,
    pub ip_address: Option<String>,
}

/// Audit log query parameters
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AuditLogQuery {
    pub entity_type: Option<String>,
    pub entity_id: Option<Uuid>,
    pub actor_id: Option<Uuid>,
    pub action: Option<String>,
    pub from_date: Option<DateTime<Utc>>,
    pub to_date: Option<DateTime<Utc>>,
    pub limit: Option<i64>,
    pub offset: Option<i64>,
}

/// Jurisdiction info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JurisdictionInfo {
    pub code: String,
    pub name: String,
    pub region: String,
    pub reporting_threshold: u64,
    pub currency: String,
    pub kyc_required: bool,
    pub travel_rule_enabled: bool,
    pub travel_rule_threshold: u64,
    pub sar_required: bool,
    pub data_retention_years: u32,
    pub active_rules: u32,
}

/// Risk thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskThresholds {
    pub jurisdiction: String,
    pub low_threshold: f64,
    pub medium_threshold: f64,
    pub high_threshold: f64,
    pub critical_threshold: f64,
    pub sar_threshold: f64,
    pub auto_reject_threshold: f64,
}

impl Default for RiskThresholds {
    fn default() -> Self {
        Self {
            jurisdiction: "DEFAULT".to_string(),
            low_threshold: 0.25,
            medium_threshold: 0.5,
            high_threshold: 0.75,
            critical_threshold: 0.9,
            sar_threshold: 0.85,
            auto_reject_threshold: 0.95,
        }
    }
}
