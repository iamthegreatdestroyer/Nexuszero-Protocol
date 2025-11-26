//! Database operations for Compliance Service

use crate::error::{ComplianceError, Result};
use crate::models::*;
use chrono::Utc;
use sqlx::PgPool;
use uuid::Uuid;

/// Repository for compliance-related database operations
pub struct ComplianceRepository;

impl ComplianceRepository {
    /// Store compliance check result
    pub async fn store_check_result(
        pool: &PgPool,
        result: &ComplianceCheckResult,
    ) -> Result<()> {
        sqlx::query!(
            r#"
            INSERT INTO compliance_checks 
                (id, transaction_id, status, risk_level, risk_score, 
                 rules_checked, rules_triggered, requires_sar, checked_at, expires_at, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            "#,
            result.id,
            result.transaction_id,
            result.status.clone() as ComplianceStatus,
            result.risk_level.clone() as RiskLevel,
            result.risk_score,
            &result.rules_checked,
            serde_json::to_value(&result.rules_triggered)?,
            result.requires_sar,
            result.checked_at,
            result.expires_at,
            serde_json::to_value(&result.metadata)?
        )
        .execute(pool)
        .await?;

        Ok(())
    }

    /// Get compliance status by transaction ID
    pub async fn get_by_transaction(
        pool: &PgPool,
        tx_id: Uuid,
    ) -> Result<ComplianceCheckResult> {
        let record = sqlx::query!(
            r#"
            SELECT id, transaction_id, status as "status: ComplianceStatus", 
                   risk_level as "risk_level: RiskLevel", risk_score,
                   rules_checked, rules_triggered, requires_sar, 
                   checked_at, expires_at, metadata
            FROM compliance_checks
            WHERE transaction_id = $1
            ORDER BY checked_at DESC
            LIMIT 1
            "#,
            tx_id
        )
        .fetch_optional(pool)
        .await?
        .ok_or_else(|| ComplianceError::TransactionNotFound(tx_id.to_string()))?;

        Ok(ComplianceCheckResult {
            id: record.id,
            transaction_id: record.transaction_id,
            status: record.status,
            risk_level: record.risk_level,
            risk_score: record.risk_score,
            rules_checked: record.rules_checked,
            rules_triggered: serde_json::from_value(record.rules_triggered)?,
            recommendations: vec![],
            requires_sar: record.requires_sar,
            checked_at: record.checked_at,
            expires_at: record.expires_at,
            metadata: serde_json::from_value(record.metadata)?,
        })
    }

    /// Get compliance history for entity
    pub async fn get_entity_history(
        pool: &PgPool,
        entity_id: Uuid,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<ComplianceCheckResult>> {
        let records = sqlx::query!(
            r#"
            SELECT cc.id, cc.transaction_id, cc.status as "status: ComplianceStatus",
                   cc.risk_level as "risk_level: RiskLevel", cc.risk_score,
                   cc.rules_checked, cc.rules_triggered, cc.requires_sar,
                   cc.checked_at, cc.expires_at, cc.metadata
            FROM compliance_checks cc
            JOIN transactions t ON cc.transaction_id = t.id
            WHERE t.sender_id = $1 OR t.recipient_id = $1
            ORDER BY cc.checked_at DESC
            LIMIT $2 OFFSET $3
            "#,
            entity_id,
            limit,
            offset
        )
        .fetch_all(pool)
        .await?;

        let mut results = Vec::with_capacity(records.len());
        for record in records {
            results.push(ComplianceCheckResult {
                id: record.id,
                transaction_id: record.transaction_id,
                status: record.status,
                risk_level: record.risk_level,
                risk_score: record.risk_score,
                rules_checked: record.rules_checked,
                rules_triggered: serde_json::from_value(record.rules_triggered)?,
                recommendations: vec![],
                requires_sar: record.requires_sar,
                checked_at: record.checked_at,
                expires_at: record.expires_at,
                metadata: serde_json::from_value(record.metadata)?,
            });
        }

        Ok(results)
    }
}

/// Repository for SAR operations
pub struct SarRepository;

impl SarRepository {
    /// Create new SAR
    pub async fn create(pool: &PgPool, sar: &SuspiciousActivityReport) -> Result<Uuid> {
        let id = sqlx::query_scalar!(
            r#"
            INSERT INTO suspicious_activity_reports
                (id, reference_number, status, subject_id, subject_type,
                 activity_type, amount_involved, currency, jurisdiction,
                 description, transaction_ids, evidence, created_at, created_by)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
            RETURNING id
            "#,
            sar.id,
            sar.reference_number,
            sar.status.clone() as SarStatus,
            sar.subject_id,
            sar.subject_type,
            sar.activity_type,
            sar.amount_involved as i64,
            sar.currency,
            sar.jurisdiction,
            sar.description,
            &sar.transaction_ids,
            serde_json::to_value(&sar.evidence)?,
            sar.created_at,
            sar.created_by
        )
        .fetch_one(pool)
        .await?;

        Ok(id)
    }

    /// Get SAR by ID
    pub async fn get_by_id(pool: &PgPool, id: Uuid) -> Result<SuspiciousActivityReport> {
        let record = sqlx::query!(
            r#"
            SELECT id, reference_number, status as "status: SarStatus",
                   subject_id, subject_type, activity_type, 
                   amount_involved, currency, jurisdiction, description,
                   transaction_ids, evidence, created_at, updated_at,
                   submitted_at, created_by, reviewed_by
            FROM suspicious_activity_reports
            WHERE id = $1
            "#,
            id
        )
        .fetch_optional(pool)
        .await?
        .ok_or_else(|| ComplianceError::SarNotFound(id.to_string()))?;

        Ok(SuspiciousActivityReport {
            id: record.id,
            reference_number: record.reference_number,
            status: record.status,
            subject_id: record.subject_id,
            subject_type: record.subject_type,
            activity_type: record.activity_type,
            amount_involved: record.amount_involved as u64,
            currency: record.currency,
            jurisdiction: record.jurisdiction,
            description: record.description,
            transaction_ids: record.transaction_ids,
            evidence: serde_json::from_value(record.evidence)?,
            created_at: record.created_at,
            updated_at: record.updated_at,
            submitted_at: record.submitted_at,
            created_by: record.created_by,
            reviewed_by: record.reviewed_by,
        })
    }

    /// List SARs with filters
    pub async fn list(
        pool: &PgPool,
        status: Option<SarStatus>,
        jurisdiction: Option<&str>,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<SuspiciousActivityReport>> {
        let records = sqlx::query!(
            r#"
            SELECT id, reference_number, status as "status: SarStatus",
                   subject_id, subject_type, activity_type,
                   amount_involved, currency, jurisdiction, description,
                   transaction_ids, evidence, created_at, updated_at,
                   submitted_at, created_by, reviewed_by
            FROM suspicious_activity_reports
            WHERE ($1::text IS NULL OR status::text = $1)
              AND ($2::text IS NULL OR jurisdiction = $2)
            ORDER BY created_at DESC
            LIMIT $3 OFFSET $4
            "#,
            status.map(|s| format!("{:?}", s).to_lowercase()),
            jurisdiction,
            limit,
            offset
        )
        .fetch_all(pool)
        .await?;

        let mut results = Vec::with_capacity(records.len());
        for record in records {
            results.push(SuspiciousActivityReport {
                id: record.id,
                reference_number: record.reference_number,
                status: record.status,
                subject_id: record.subject_id,
                subject_type: record.subject_type,
                activity_type: record.activity_type,
                amount_involved: record.amount_involved as u64,
                currency: record.currency,
                jurisdiction: record.jurisdiction,
                description: record.description,
                transaction_ids: record.transaction_ids,
                evidence: serde_json::from_value(record.evidence)?,
                created_at: record.created_at,
                updated_at: record.updated_at,
                submitted_at: record.submitted_at,
                created_by: record.created_by,
                reviewed_by: record.reviewed_by,
            });
        }

        Ok(results)
    }

    /// Update SAR status
    pub async fn update_status(
        pool: &PgPool,
        id: Uuid,
        status: SarStatus,
        reviewed_by: Option<Uuid>,
    ) -> Result<()> {
        let submitted_at = if status == SarStatus::Submitted {
            Some(Utc::now())
        } else {
            None
        };

        sqlx::query!(
            r#"
            UPDATE suspicious_activity_reports
            SET status = $2, reviewed_by = $3, submitted_at = COALESCE($4, submitted_at),
                updated_at = NOW()
            WHERE id = $1
            "#,
            id,
            status as SarStatus,
            reviewed_by,
            submitted_at
        )
        .execute(pool)
        .await?;

        Ok(())
    }
}

/// Repository for rule operations
pub struct RuleRepository;

impl RuleRepository {
    /// Create new rule
    pub async fn create(pool: &PgPool, rule: &ComplianceRule) -> Result<Uuid> {
        let id = sqlx::query_scalar!(
            r#"
            INSERT INTO compliance_rules
                (id, name, description, rule_type, jurisdiction, severity,
                 conditions, actions, is_active, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $10)
            RETURNING id
            "#,
            rule.id,
            rule.name,
            rule.description,
            rule.rule_type.clone() as RuleType,
            rule.jurisdiction,
            rule.severity.clone() as RiskLevel,
            rule.conditions,
            &rule.actions,
            rule.is_active,
            rule.created_at
        )
        .fetch_one(pool)
        .await?;

        Ok(id)
    }

    /// Get rule by ID
    pub async fn get_by_id(pool: &PgPool, id: Uuid) -> Result<ComplianceRule> {
        let record = sqlx::query!(
            r#"
            SELECT id, name, description, rule_type as "rule_type: RuleType",
                   jurisdiction, severity as "severity: RiskLevel",
                   conditions, actions, is_active, created_at, updated_at
            FROM compliance_rules
            WHERE id = $1
            "#,
            id
        )
        .fetch_optional(pool)
        .await?
        .ok_or_else(|| ComplianceError::RuleNotFound(id.to_string()))?;

        Ok(ComplianceRule {
            id: record.id,
            name: record.name,
            description: record.description,
            rule_type: record.rule_type,
            jurisdiction: record.jurisdiction,
            severity: record.severity,
            conditions: record.conditions,
            actions: record.actions,
            is_active: record.is_active,
            created_at: record.created_at,
            updated_at: record.updated_at,
        })
    }

    /// List rules by jurisdiction
    pub async fn list_by_jurisdiction(
        pool: &PgPool,
        jurisdiction: &str,
        active_only: bool,
    ) -> Result<Vec<ComplianceRule>> {
        let records = sqlx::query!(
            r#"
            SELECT id, name, description, rule_type as "rule_type: RuleType",
                   jurisdiction, severity as "severity: RiskLevel",
                   conditions, actions, is_active, created_at, updated_at
            FROM compliance_rules
            WHERE jurisdiction = $1 AND ($2 = false OR is_active = true)
            ORDER BY severity DESC, name ASC
            "#,
            jurisdiction,
            active_only
        )
        .fetch_all(pool)
        .await?;

        Ok(records
            .into_iter()
            .map(|r| ComplianceRule {
                id: r.id,
                name: r.name,
                description: r.description,
                rule_type: r.rule_type,
                jurisdiction: r.jurisdiction,
                severity: r.severity,
                conditions: r.conditions,
                actions: r.actions,
                is_active: r.is_active,
                created_at: r.created_at,
                updated_at: r.updated_at,
            })
            .collect())
    }

    /// Update rule
    pub async fn update(pool: &PgPool, rule: &ComplianceRule) -> Result<()> {
        sqlx::query!(
            r#"
            UPDATE compliance_rules
            SET name = $2, description = $3, rule_type = $4, severity = $5,
                conditions = $6, actions = $7, is_active = $8, updated_at = NOW()
            WHERE id = $1
            "#,
            rule.id,
            rule.name,
            rule.description,
            rule.rule_type.clone() as RuleType,
            rule.severity.clone() as RiskLevel,
            rule.conditions,
            &rule.actions,
            rule.is_active
        )
        .execute(pool)
        .await?;

        Ok(())
    }

    /// Delete rule
    pub async fn delete(pool: &PgPool, id: Uuid) -> Result<()> {
        sqlx::query!("DELETE FROM compliance_rules WHERE id = $1", id)
            .execute(pool)
            .await?;
        Ok(())
    }
}

/// Repository for audit log operations
pub struct AuditRepository;

impl AuditRepository {
    /// Insert audit log entry
    pub async fn log(pool: &PgPool, entry: &AuditLogEntry) -> Result<()> {
        sqlx::query!(
            r#"
            INSERT INTO audit_logs
                (id, timestamp, action, entity_type, entity_id,
                 actor_id, actor_type, details, ip_address)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            "#,
            entry.id,
            entry.timestamp,
            entry.action,
            entry.entity_type,
            entry.entity_id,
            entry.actor_id,
            entry.actor_type,
            entry.details,
            entry.ip_address
        )
        .execute(pool)
        .await?;

        Ok(())
    }

    /// Query audit logs
    pub async fn query(pool: &PgPool, query: &AuditLogQuery) -> Result<Vec<AuditLogEntry>> {
        let records = sqlx::query!(
            r#"
            SELECT id, timestamp, action, entity_type, entity_id,
                   actor_id, actor_type, details, ip_address
            FROM audit_logs
            WHERE ($1::text IS NULL OR entity_type = $1)
              AND ($2::uuid IS NULL OR entity_id = $2)
              AND ($3::uuid IS NULL OR actor_id = $3)
              AND ($4::text IS NULL OR action = $4)
              AND ($5::timestamptz IS NULL OR timestamp >= $5)
              AND ($6::timestamptz IS NULL OR timestamp <= $6)
            ORDER BY timestamp DESC
            LIMIT $7 OFFSET $8
            "#,
            query.entity_type,
            query.entity_id,
            query.actor_id,
            query.action,
            query.from_date,
            query.to_date,
            query.limit.unwrap_or(100),
            query.offset.unwrap_or(0)
        )
        .fetch_all(pool)
        .await?;

        Ok(records
            .into_iter()
            .map(|r| AuditLogEntry {
                id: r.id,
                timestamp: r.timestamp,
                action: r.action,
                entity_type: r.entity_type,
                entity_id: r.entity_id,
                actor_id: r.actor_id,
                actor_type: r.actor_type,
                details: r.details,
                ip_address: r.ip_address,
            })
            .collect())
    }
}
