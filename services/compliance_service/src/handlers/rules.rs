//! Rules management handlers

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
use crate::rules::RuleEvaluator;
use crate::state::AppState;

#[derive(Debug, Deserialize)]
pub struct RuleListParams {
    pub jurisdiction: Option<String>,
    pub rule_type: Option<String>,
    pub active_only: Option<bool>,
}

/// List compliance rules
pub async fn list_rules(
    State(state): State<Arc<AppState>>,
    Query(params): Query<RuleListParams>,
) -> Json<Vec<ComplianceRule>> {
    let engine = state.rule_engine.read().await;
    
    if let Some(jurisdiction) = params.jurisdiction {
        Json(engine.get_rules(&jurisdiction))
    } else {
        // Return all rules
        let mut all_rules = Vec::new();
        for jurisdiction in &state.config.enabled_jurisdictions {
            all_rules.extend(engine.get_rules(jurisdiction));
        }
        Json(all_rules)
    }
}

/// Create new rule
pub async fn create_rule(
    State(state): State<Arc<AppState>>,
    Json(request): Json<CreateRuleRequest>,
) -> Result<(StatusCode, Json<ComplianceRule>)> {
    let rule = ComplianceRule {
        id: Uuid::new_v4(),
        name: request.name,
        description: request.description,
        rule_type: request.rule_type,
        jurisdiction: request.jurisdiction,
        severity: request.severity,
        conditions: request.conditions,
        actions: request.actions,
        is_active: true,
        created_at: Utc::now(),
        updated_at: Utc::now(),
    };
    
    // Add to engine
    let mut engine = state.rule_engine.write().await;
    engine.add_rule(rule.clone());
    
    Ok((StatusCode::CREATED, Json(rule)))
}

/// Get rule by ID
pub async fn get_rule(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Json<ComplianceRule>> {
    let engine = state.rule_engine.read().await;
    
    for jurisdiction in &state.config.enabled_jurisdictions {
        let rules = engine.get_rules(jurisdiction);
        if let Some(rule) = rules.into_iter().find(|r| r.id == id) {
            return Ok(Json(rule));
        }
    }
    
    Err(ComplianceError::RuleNotFound(id.to_string()))
}

/// Update rule
pub async fn update_rule(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Json(request): Json<CreateRuleRequest>,
) -> Result<Json<ComplianceRule>> {
    let rule = ComplianceRule {
        id,
        name: request.name,
        description: request.description,
        rule_type: request.rule_type,
        jurisdiction: request.jurisdiction.clone(),
        severity: request.severity,
        conditions: request.conditions,
        actions: request.actions,
        is_active: true,
        created_at: Utc::now(), // Would preserve original in production
        updated_at: Utc::now(),
    };
    
    let mut engine = state.rule_engine.write().await;
    
    // Remove old rule
    engine.remove_rule(&request.jurisdiction, id);
    
    // Add updated rule
    engine.add_rule(rule.clone());
    
    Ok(Json(rule))
}

/// Delete rule
pub async fn delete_rule(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<StatusCode> {
    let mut engine = state.rule_engine.write().await;
    
    for jurisdiction in &state.config.enabled_jurisdictions {
        if engine.remove_rule(jurisdiction, id) {
            return Ok(StatusCode::NO_CONTENT);
        }
    }
    
    Err(ComplianceError::RuleNotFound(id.to_string()))
}

/// Test rule against sample data
pub async fn test_rule(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Json(request): Json<RuleTestRequest>,
) -> Result<Json<RuleTestResult>> {
    let engine = state.rule_engine.read().await;
    
    // Find the rule
    let mut found_rule = None;
    for jurisdiction in &state.config.enabled_jurisdictions {
        let rules = engine.get_rules(jurisdiction);
        if let Some(rule) = rules.into_iter().find(|r| r.id == id) {
            found_rule = Some(rule);
            break;
        }
    }
    
    let rule = found_rule
        .ok_or_else(|| ComplianceError::RuleNotFound(id.to_string()))?;
    
    // Test the rule
    let mut evaluator = RuleEvaluator::new();
    let result = evaluator.test_rule(&rule, &request.test_data)?;
    
    Ok(Json(result))
}
