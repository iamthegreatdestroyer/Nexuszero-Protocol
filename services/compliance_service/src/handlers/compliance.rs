//! Compliance checking handlers

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Json,
};
use serde::Deserialize;
use std::sync::Arc;
use uuid::Uuid;

use crate::error::Result;
use crate::handlers::metrics::{COMPLIANCE_CHECKS_TOTAL, COMPLIANCE_CHECK_DURATION};
use crate::models::*;
use crate::state::AppState;

/// Check single transaction for compliance
pub async fn check_transaction(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ComplianceCheckRequest>,
) -> Result<(StatusCode, Json<ComplianceCheckResult>)> {
    let timer = COMPLIANCE_CHECK_DURATION.start_timer();
    
    // Evaluate against rules
    let result = {
        let engine = state.rule_engine.read().await;
        engine.evaluate(&request)
    };
    
    // Record metrics
    COMPLIANCE_CHECKS_TOTAL
        .with_label_values(&[
            &request.jurisdiction,
            &format!("{:?}", result.status).to_lowercase(),
        ])
        .inc();
    
    timer.observe_duration();
    
    Ok((StatusCode::OK, Json(result)))
}

/// Check batch of transactions
pub async fn check_batch(
    State(state): State<Arc<AppState>>,
    Json(request): Json<BatchComplianceRequest>,
) -> Result<(StatusCode, Json<BatchComplianceResult>)> {
    let start = std::time::Instant::now();
    
    let mut results = Vec::with_capacity(request.transactions.len());
    let mut approved = 0;
    let mut rejected = 0;
    let mut requires_review = 0;
    
    let engine = state.rule_engine.read().await;
    
    for tx in &request.transactions {
        let result = engine.evaluate(tx);
        
        match result.status {
            ComplianceStatus::Approved => approved += 1,
            ComplianceStatus::Rejected => rejected += 1,
            ComplianceStatus::RequiresReview | ComplianceStatus::Escalated => requires_review += 1,
            _ => {}
        }
        
        // Record metrics
        COMPLIANCE_CHECKS_TOTAL
            .with_label_values(&[
                &tx.jurisdiction,
                &format!("{:?}", result.status).to_lowercase(),
            ])
            .inc();
        
        results.push(result);
    }
    
    let elapsed = start.elapsed();
    
    let batch_result = BatchComplianceResult {
        batch_id: Uuid::new_v4(),
        results,
        summary: BatchSummary {
            total: request.transactions.len(),
            approved,
            rejected,
            requires_review,
            processing_time_ms: elapsed.as_millis() as u64,
        },
    };
    
    Ok((StatusCode::OK, Json(batch_result)))
}

#[derive(Debug, Deserialize)]
pub struct StatusParams {
    pub include_history: Option<bool>,
}

/// Get compliance status for transaction
pub async fn get_status(
    State(_state): State<Arc<AppState>>,
    Path(tx_id): Path<Uuid>,
    Query(_params): Query<StatusParams>,
) -> Result<Json<ComplianceCheckResult>> {
    // In production, this would query the database
    // For now, return a placeholder
    Ok(Json(ComplianceCheckResult {
        id: Uuid::new_v4(),
        transaction_id: tx_id,
        status: ComplianceStatus::Approved,
        risk_level: RiskLevel::Low,
        risk_score: 0.15,
        rules_checked: vec!["threshold".to_string(), "velocity".to_string()],
        rules_triggered: vec![],
        recommendations: vec![],
        requires_sar: false,
        checked_at: chrono::Utc::now(),
        expires_at: Some(chrono::Utc::now() + chrono::Duration::days(30)),
        metadata: std::collections::HashMap::new(),
    }))
}

#[derive(Debug, Deserialize)]
pub struct HistoryParams {
    pub limit: Option<i64>,
    pub offset: Option<i64>,
}

/// Get compliance history for entity
pub async fn get_history(
    State(_state): State<Arc<AppState>>,
    Path(entity_id): Path<Uuid>,
    Query(params): Query<HistoryParams>,
) -> Result<Json<Vec<ComplianceCheckResult>>> {
    let _limit = params.limit.unwrap_or(20);
    let _offset = params.offset.unwrap_or(0);
    
    // In production, this would query the database
    // For now, return empty history
    Ok(Json(vec![]))
}
