//! Compliance Checker Service

use crate::models::*;
use crate::rules::RuleEngine;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Service for compliance checking operations
pub struct ComplianceChecker {
    rule_engine: Arc<RwLock<RuleEngine>>,
}

impl ComplianceChecker {
    pub fn new(rule_engine: Arc<RwLock<RuleEngine>>) -> Self {
        Self { rule_engine }
    }

    /// Check single transaction
    pub async fn check(&self, request: &ComplianceCheckRequest) -> ComplianceCheckResult {
        let engine = self.rule_engine.read().await;
        engine.evaluate(request)
    }

    /// Check batch of transactions
    pub async fn check_batch(&self, requests: &[ComplianceCheckRequest]) -> Vec<ComplianceCheckResult> {
        let engine = self.rule_engine.read().await;
        requests.iter().map(|req| engine.evaluate(req)).collect()
    }

    /// Check if transaction requires SAR
    pub fn requires_sar(&self, result: &ComplianceCheckResult, threshold: f64) -> bool {
        result.risk_score >= threshold
    }
}
