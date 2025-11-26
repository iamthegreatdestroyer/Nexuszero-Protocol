//! Report generation handlers

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Json,
};
use chrono::Utc;
use serde::Deserialize;
use std::sync::Arc;
use uuid::Uuid;

use crate::error::{ComplianceError, Result};
use crate::models::*;
use crate::state::AppState;

#[derive(Debug, Deserialize)]
pub struct ReportListParams {
    pub report_type: Option<String>,
    pub jurisdiction: Option<String>,
    pub limit: Option<i64>,
    pub offset: Option<i64>,
}

/// List generated reports
pub async fn list_reports(
    State(_state): State<Arc<AppState>>,
    Query(params): Query<ReportListParams>,
) -> Json<Vec<ComplianceReport>> {
    // In production, this would query the database
    Json(vec![])
}

/// Get specific report
pub async fn get_report(
    State(_state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Json<ComplianceReport>> {
    Err(ComplianceError::ReportNotFound(id.to_string()))
}

/// Generate new report
pub async fn generate_report(
    State(_state): State<Arc<AppState>>,
    Json(request): Json<GenerateReportRequest>,
) -> Result<(StatusCode, Json<ComplianceReport>)> {
    let report_id = Uuid::new_v4();
    
    // In production, this would trigger async report generation
    let summary = generate_report_summary(&request).await;
    
    let report = ComplianceReport {
        id: report_id,
        report_type: request.report_type,
        jurisdiction: request.jurisdiction,
        period_start: request.period_start,
        period_end: request.period_end,
        generated_at: Utc::now(),
        summary,
        status: "completed".to_string(),
        file_url: Some(format!("/reports/{}/download", report_id)),
    };
    
    Ok((StatusCode::CREATED, Json(report)))
}

/// Schedule recurring report
pub async fn schedule_report(
    State(_state): State<Arc<AppState>>,
    Json(request): Json<ScheduleReportRequest>,
) -> Result<(StatusCode, Json<ReportSchedule>)> {
    let schedule_id = Uuid::new_v4();
    
    let schedule = ReportSchedule {
        id: schedule_id,
        report_type: request.report_type,
        jurisdiction: request.jurisdiction,
        cron_expression: request.cron_expression,
        recipients: request.recipients,
        is_active: true,
        created_at: Utc::now(),
        next_run: calculate_next_run(&schedule_id.to_string()),
    };
    
    Ok((StatusCode::CREATED, Json(schedule)))
}

/// Report schedule
#[derive(Debug, serde::Serialize)]
pub struct ReportSchedule {
    pub id: Uuid,
    pub report_type: ReportType,
    pub jurisdiction: String,
    pub cron_expression: String,
    pub recipients: Vec<String>,
    pub is_active: bool,
    pub created_at: chrono::DateTime<Utc>,
    pub next_run: chrono::DateTime<Utc>,
}

/// Generate report summary (simulated)
async fn generate_report_summary(request: &GenerateReportRequest) -> ReportSummary {
    // In production, this would query actual data
    ReportSummary {
        total_transactions: 15000,
        transactions_approved: 14500,
        transactions_rejected: 200,
        sars_filed: 25,
        high_risk_entities: 150,
        watchlist_matches: 10,
        average_risk_score: 0.22,
    }
}

/// Calculate next run time from cron expression
fn calculate_next_run(cron_expr: &str) -> chrono::DateTime<Utc> {
    // Simplified - in production would parse cron expression
    Utc::now() + chrono::Duration::days(1)
}
