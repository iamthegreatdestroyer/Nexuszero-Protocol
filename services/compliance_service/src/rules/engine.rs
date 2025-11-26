//! Rule Engine - Core compliance rule processing

use crate::config::ComplianceConfig;
use crate::models::*;
use std::collections::HashMap;
use uuid::Uuid;
use chrono::Utc;

/// Rule engine for evaluating compliance rules
pub struct RuleEngine {
    /// Active rules by jurisdiction
    rules: HashMap<String, Vec<ComplianceRule>>,
    /// Risk thresholds by jurisdiction
    thresholds: HashMap<String, RiskThresholds>,
    /// Configuration
    config: ComplianceConfig,
}

impl RuleEngine {
    /// Create new rule engine
    pub fn new(config: &ComplianceConfig) -> Self {
        let mut engine = Self {
            rules: HashMap::new(),
            thresholds: HashMap::new(),
            config: config.clone(),
        };
        
        // Initialize default rules for enabled jurisdictions
        for jurisdiction in &config.enabled_jurisdictions {
            engine.rules.insert(jurisdiction.clone(), Self::default_rules(jurisdiction));
            engine.thresholds.insert(jurisdiction.clone(), RiskThresholds {
                jurisdiction: jurisdiction.clone(),
                ..Default::default()
            });
        }
        
        engine
    }

    /// Evaluate transaction against all applicable rules
    pub fn evaluate(&self, request: &ComplianceCheckRequest) -> ComplianceCheckResult {
        let jurisdiction = &request.jurisdiction;
        let rules = self.rules.get(jurisdiction).cloned().unwrap_or_default();
        let thresholds = self.thresholds.get(jurisdiction)
            .cloned()
            .unwrap_or_default();
        
        let mut rules_checked = Vec::new();
        let mut rules_triggered = Vec::new();
        let mut risk_score: f64 = 0.0;
        let mut total_weight: f64 = 0.0;

        // Evaluate each rule
        for rule in &rules {
            if !rule.is_active {
                continue;
            }
            
            rules_checked.push(rule.name.clone());
            
            if let Some(violation) = self.evaluate_rule(rule, request) {
                let severity_weight = match violation.severity {
                    RiskLevel::Low => 0.1,
                    RiskLevel::Medium => 0.25,
                    RiskLevel::High => 0.5,
                    RiskLevel::Critical => 0.8,
                    RiskLevel::Blocked => 1.0,
                };
                
                risk_score += severity_weight;
                total_weight += 1.0;
                rules_triggered.push(violation);
            } else {
                total_weight += 1.0;
            }
        }

        // Normalize risk score
        let final_risk_score: f64 = if total_weight > 0.0 {
            (risk_score / total_weight).min(1.0)
        } else {
            0.0
        };

        // Determine status based on thresholds
        let status = if final_risk_score >= thresholds.auto_reject_threshold {
            ComplianceStatus::Rejected
        } else if final_risk_score >= thresholds.critical_threshold {
            ComplianceStatus::Escalated
        } else if final_risk_score >= thresholds.high_threshold {
            ComplianceStatus::RequiresReview
        } else {
            ComplianceStatus::Approved
        };

        let risk_level = RiskLevel::from_score(final_risk_score);
        let requires_sar = final_risk_score >= thresholds.sar_threshold;

        // Generate recommendations
        let recommendations = self.generate_recommendations(&rules_triggered, &status);

        ComplianceCheckResult {
            id: Uuid::new_v4(),
            transaction_id: request.transaction_id,
            status,
            risk_level,
            risk_score: final_risk_score,
            rules_checked,
            rules_triggered,
            recommendations,
            requires_sar,
            checked_at: Utc::now(),
            expires_at: Some(Utc::now() + chrono::Duration::days(30)),
            metadata: request.metadata.clone(),
        }
    }

    /// Evaluate single rule against request
    fn evaluate_rule(
        &self,
        rule: &ComplianceRule,
        request: &ComplianceCheckRequest,
    ) -> Option<RuleViolation> {
        match rule.rule_type {
            RuleType::Threshold => self.evaluate_threshold_rule(rule, request),
            RuleType::Pattern => self.evaluate_pattern_rule(rule, request),
            RuleType::Velocity => self.evaluate_velocity_rule(rule, request),
            RuleType::Watchlist => self.evaluate_watchlist_rule(rule, request),
            RuleType::Geographic => self.evaluate_geographic_rule(rule, request),
            RuleType::Behavioral => self.evaluate_behavioral_rule(rule, request),
            RuleType::Custom => self.evaluate_custom_rule(rule, request),
        }
    }

    /// Evaluate threshold-based rule
    fn evaluate_threshold_rule(
        &self,
        rule: &ComplianceRule,
        request: &ComplianceCheckRequest,
    ) -> Option<RuleViolation> {
        let threshold = rule.conditions.get("threshold")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        
        if request.amount >= threshold {
            Some(RuleViolation {
                rule_id: rule.id.to_string(),
                rule_name: rule.name.clone(),
                severity: rule.severity.clone(),
                description: format!(
                    "Transaction amount {} exceeds threshold {}",
                    request.amount, threshold
                ),
                jurisdiction: request.jurisdiction.clone(),
                remediation: Some("Verify source of funds and perform enhanced due diligence".to_string()),
            })
        } else {
            None
        }
    }

    /// Evaluate pattern-based rule
    fn evaluate_pattern_rule(
        &self,
        rule: &ComplianceRule,
        _request: &ComplianceCheckRequest,
    ) -> Option<RuleViolation> {
        // Pattern detection would analyze transaction history
        // Placeholder for pattern matching logic
        let _patterns = rule.conditions.get("patterns")
            .and_then(|v| v.as_array());
        
        // TODO: Implement pattern matching against historical data
        None
    }

    /// Evaluate velocity-based rule
    fn evaluate_velocity_rule(
        &self,
        rule: &ComplianceRule,
        _request: &ComplianceCheckRequest,
    ) -> Option<RuleViolation> {
        // Velocity checks would query recent transaction counts
        let _time_window = rule.conditions.get("time_window_hours")
            .and_then(|v| v.as_u64())
            .unwrap_or(24);
        let _max_count = rule.conditions.get("max_transactions")
            .and_then(|v| v.as_u64())
            .unwrap_or(100);
        
        // TODO: Implement velocity checking
        None
    }

    /// Evaluate watchlist rule
    fn evaluate_watchlist_rule(
        &self,
        rule: &ComplianceRule,
        _request: &ComplianceCheckRequest,
    ) -> Option<RuleViolation> {
        // Watchlist checking would query external services
        let _watchlists = rule.conditions.get("watchlists")
            .and_then(|v| v.as_array());
        
        // TODO: Implement watchlist checking
        None
    }

    /// Evaluate geographic rule
    fn evaluate_geographic_rule(
        &self,
        rule: &ComplianceRule,
        request: &ComplianceCheckRequest,
    ) -> Option<RuleViolation> {
        let blocked_countries = rule.conditions.get("blocked_countries")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter()
                .filter_map(|v| v.as_str())
                .map(String::from)
                .collect::<Vec<_>>())
            .unwrap_or_default();
        
        let country = request.metadata.get("country").cloned().unwrap_or_default();
        
        if blocked_countries.contains(&country) {
            Some(RuleViolation {
                rule_id: rule.id.to_string(),
                rule_name: rule.name.clone(),
                severity: RiskLevel::Critical,
                description: format!("Transaction from blocked country: {}", country),
                jurisdiction: request.jurisdiction.clone(),
                remediation: Some("Transaction from sanctioned jurisdiction not permitted".to_string()),
            })
        } else {
            None
        }
    }

    /// Evaluate behavioral rule
    fn evaluate_behavioral_rule(
        &self,
        rule: &ComplianceRule,
        _request: &ComplianceCheckRequest,
    ) -> Option<RuleViolation> {
        // Behavioral analysis would use ML models
        let _indicators = rule.conditions.get("indicators")
            .and_then(|v| v.as_array());
        
        // TODO: Implement behavioral analysis
        None
    }

    /// Evaluate custom rule
    fn evaluate_custom_rule(
        &self,
        rule: &ComplianceRule,
        _request: &ComplianceCheckRequest,
    ) -> Option<RuleViolation> {
        // Custom rules use embedded scripts or expressions
        let _expression = rule.conditions.get("expression")
            .and_then(|v| v.as_str());
        
        // TODO: Implement custom expression evaluation
        None
    }

    /// Generate recommendations based on violations
    fn generate_recommendations(
        &self,
        violations: &[RuleViolation],
        status: &ComplianceStatus,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        if violations.is_empty() {
            recommendations.push("No compliance issues detected".to_string());
            return recommendations;
        }

        match status {
            ComplianceStatus::Rejected => {
                recommendations.push("Transaction blocked due to compliance violations".to_string());
                recommendations.push("Manual review by compliance officer required".to_string());
            }
            ComplianceStatus::Escalated => {
                recommendations.push("Transaction requires senior compliance review".to_string());
                recommendations.push("Document all supporting evidence before approval".to_string());
            }
            ComplianceStatus::RequiresReview => {
                recommendations.push("Transaction flagged for compliance review".to_string());
                recommendations.push("Verify customer identity and source of funds".to_string());
            }
            _ => {}
        }

        // Add violation-specific recommendations
        for violation in violations {
            if let Some(ref remediation) = violation.remediation {
                if !recommendations.contains(remediation) {
                    recommendations.push(remediation.clone());
                }
            }
        }

        recommendations
    }

    /// Default rules for a jurisdiction
    fn default_rules(jurisdiction: &str) -> Vec<ComplianceRule> {
        let now = Utc::now();
        
        vec![
            // Large transaction threshold
            ComplianceRule {
                id: Uuid::new_v4(),
                name: "Large Transaction Threshold".to_string(),
                description: "Flag transactions exceeding reporting threshold".to_string(),
                rule_type: RuleType::Threshold,
                jurisdiction: jurisdiction.to_string(),
                severity: RiskLevel::Medium,
                conditions: serde_json::json!({
                    "threshold": 10000,
                    "currency": "USD"
                }),
                actions: vec!["flag".to_string(), "report".to_string()],
                is_active: true,
                created_at: now,
                updated_at: now,
            },
            // Very large transaction
            ComplianceRule {
                id: Uuid::new_v4(),
                name: "Very Large Transaction".to_string(),
                description: "Escalate very large transactions".to_string(),
                rule_type: RuleType::Threshold,
                jurisdiction: jurisdiction.to_string(),
                severity: RiskLevel::High,
                conditions: serde_json::json!({
                    "threshold": 100000,
                    "currency": "USD"
                }),
                actions: vec!["escalate".to_string(), "sar_required".to_string()],
                is_active: true,
                created_at: now,
                updated_at: now,
            },
            // Sanctioned countries
            ComplianceRule {
                id: Uuid::new_v4(),
                name: "Sanctioned Countries".to_string(),
                description: "Block transactions from OFAC sanctioned countries".to_string(),
                rule_type: RuleType::Geographic,
                jurisdiction: jurisdiction.to_string(),
                severity: RiskLevel::Blocked,
                conditions: serde_json::json!({
                    "blocked_countries": ["KP", "IR", "SY", "CU"]
                }),
                actions: vec!["block".to_string(), "alert".to_string()],
                is_active: true,
                created_at: now,
                updated_at: now,
            },
            // High velocity
            ComplianceRule {
                id: Uuid::new_v4(),
                name: "Transaction Velocity".to_string(),
                description: "Flag unusual transaction velocity".to_string(),
                rule_type: RuleType::Velocity,
                jurisdiction: jurisdiction.to_string(),
                severity: RiskLevel::Medium,
                conditions: serde_json::json!({
                    "time_window_hours": 24,
                    "max_transactions": 50,
                    "max_amount": 50000
                }),
                actions: vec!["flag".to_string(), "review".to_string()],
                is_active: true,
                created_at: now,
                updated_at: now,
            },
        ]
    }

    /// Add a new rule
    pub fn add_rule(&mut self, rule: ComplianceRule) {
        let jurisdiction = rule.jurisdiction.clone();
        self.rules
            .entry(jurisdiction)
            .or_insert_with(Vec::new)
            .push(rule);
    }

    /// Get rules for jurisdiction
    pub fn get_rules(&self, jurisdiction: &str) -> Vec<ComplianceRule> {
        self.rules.get(jurisdiction).cloned().unwrap_or_default()
    }

    /// Get thresholds for jurisdiction
    pub fn get_thresholds(&self, jurisdiction: &str) -> RiskThresholds {
        self.thresholds.get(jurisdiction).cloned().unwrap_or_default()
    }

    /// Update thresholds
    pub fn update_thresholds(&mut self, thresholds: RiskThresholds) {
        let jurisdiction = thresholds.jurisdiction.clone();
        self.thresholds.insert(jurisdiction, thresholds);
    }

    /// Remove a rule
    pub fn remove_rule(&mut self, jurisdiction: &str, rule_id: Uuid) -> bool {
        if let Some(rules) = self.rules.get_mut(jurisdiction) {
            let original_len = rules.len();
            rules.retain(|r| r.id != rule_id);
            return rules.len() < original_len;
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn test_config() -> ComplianceConfig {
        ComplianceConfig {
            port: 8083,
            database_url: "postgres://localhost/test".to_string(),
            redis_url: "redis://localhost".to_string(),
            enabled_jurisdictions: ["FATF", "US"].iter().map(|s| s.to_string()).collect(),
            default_risk_threshold: 0.7,
            sar_threshold: 0.85,
            travel_rule_threshold: 3000,
            aml_service_url: None,
            kyc_service_url: None,
            watchlist_update_interval: 3600,
            report_workers: 4,
            realtime_monitoring: true,
            max_batch_size: 100,
            audit_retention_days: 2555,
        }
    }

    #[test]
    fn test_rule_engine_creation() {
        let config = test_config();
        let engine = RuleEngine::new(&config);
        
        assert!(!engine.rules.is_empty());
        assert!(engine.rules.contains_key("FATF"));
        assert!(engine.rules.contains_key("US"));
    }

    #[test]
    fn test_threshold_evaluation() {
        let config = test_config();
        let engine = RuleEngine::new(&config);
        
        // Small transaction - should pass
        let small_tx = ComplianceCheckRequest {
            transaction_id: Uuid::new_v4(),
            sender_id: Uuid::new_v4(),
            recipient_id: Uuid::new_v4(),
            amount: 1000,
            currency: "USD".to_string(),
            transaction_type: "transfer".to_string(),
            jurisdiction: "US".to_string(),
            metadata: HashMap::new(),
        };
        
        let result = engine.evaluate(&small_tx);
        assert!(matches!(result.status, ComplianceStatus::Approved));
        
        // Large transaction - should flag
        let large_tx = ComplianceCheckRequest {
            amount: 15000,
            ..small_tx.clone()
        };
        
        let result = engine.evaluate(&large_tx);
        assert!(!result.rules_triggered.is_empty());
    }

    #[test]
    fn test_geographic_blocking() {
        let config = test_config();
        let engine = RuleEngine::new(&config);
        
        let mut metadata = HashMap::new();
        metadata.insert("country".to_string(), "KP".to_string()); // North Korea
        
        let request = ComplianceCheckRequest {
            transaction_id: Uuid::new_v4(),
            sender_id: Uuid::new_v4(),
            recipient_id: Uuid::new_v4(),
            amount: 100,
            currency: "USD".to_string(),
            transaction_type: "transfer".to_string(),
            jurisdiction: "US".to_string(),
            metadata,
        };
        
        let result = engine.evaluate(&request);
        
        // Should have geographic violation
        let has_geo_violation = result.rules_triggered
            .iter()
            .any(|v| v.rule_name.contains("Sanctioned"));
        assert!(has_geo_violation);
    }

    #[test]
    fn test_risk_level_from_score() {
        assert!(matches!(RiskLevel::from_score(0.1), RiskLevel::Low));
        assert!(matches!(RiskLevel::from_score(0.4), RiskLevel::Medium));
        assert!(matches!(RiskLevel::from_score(0.6), RiskLevel::High));
        assert!(matches!(RiskLevel::from_score(0.85), RiskLevel::Critical));
        assert!(matches!(RiskLevel::from_score(0.95), RiskLevel::Blocked));
    }

    #[test]
    fn test_add_remove_rule() {
        let config = test_config();
        let mut engine = RuleEngine::new(&config);
        
        let rule = ComplianceRule {
            id: Uuid::new_v4(),
            name: "Test Rule".to_string(),
            description: "Test".to_string(),
            rule_type: RuleType::Threshold,
            jurisdiction: "US".to_string(),
            severity: RiskLevel::Low,
            conditions: serde_json::json!({}),
            actions: vec![],
            is_active: true,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        
        let rule_id = rule.id;
        engine.add_rule(rule);
        
        let rules = engine.get_rules("US");
        assert!(rules.iter().any(|r| r.id == rule_id));
        
        let removed = engine.remove_rule("US", rule_id);
        assert!(removed);
        
        let rules = engine.get_rules("US");
        assert!(!rules.iter().any(|r| r.id == rule_id));
    }
}
