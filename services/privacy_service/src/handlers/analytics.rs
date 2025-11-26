//! Analytics handlers

use crate::error::{PrivacyError, Result};
use crate::state::AppState;
use axum::{
    extract::{Extension, Query},
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Analytics query parameters
#[derive(Debug, Deserialize)]
pub struct AnalyticsQuery {
    pub start: Option<chrono::DateTime<chrono::Utc>>,
    pub end: Option<chrono::DateTime<chrono::Utc>>,
}

/// Privacy analytics response
#[derive(Debug, Serialize)]
pub struct PrivacyAnalytics {
    pub period_start: chrono::DateTime<chrono::Utc>,
    pub period_end: chrono::DateTime<chrono::Utc>,
    pub morphs: MorphStats,
    pub disclosures: DisclosureStats,
    pub level_distribution: Vec<LevelStats>,
}

/// Morph statistics
#[derive(Debug, Serialize)]
pub struct MorphStats {
    pub total: i64,
    pub upgrades: i64,
    pub downgrades: i64,
    pub average_from_level: f64,
    pub average_to_level: f64,
}

/// Disclosure statistics
#[derive(Debug, Serialize)]
pub struct DisclosureStats {
    pub total_created: i64,
    pub total_verified: i64,
    pub total_revoked: i64,
    pub active: i64,
    pub expired: i64,
}

/// Level distribution statistics
#[derive(Debug, Serialize)]
pub struct LevelStats {
    pub level: i16,
    pub name: String,
    pub count: i64,
    pub percentage: f64,
}

/// Proof analytics response
#[derive(Debug, Serialize)]
pub struct ProofAnalytics {
    pub period_start: chrono::DateTime<chrono::Utc>,
    pub period_end: chrono::DateTime<chrono::Utc>,
    pub total_generated: i64,
    pub by_status: Vec<StatusStats>,
    pub by_priority: Vec<PriorityStats>,
    pub by_level: Vec<LevelProofStats>,
    pub performance: PerformanceStats,
    pub queue: QueueAnalytics,
}

/// Status statistics
#[derive(Debug, Serialize)]
pub struct StatusStats {
    pub status: String,
    pub count: i64,
    pub percentage: f64,
}

/// Priority statistics
#[derive(Debug, Serialize)]
pub struct PriorityStats {
    pub priority: String,
    pub count: i64,
    pub avg_wait_ms: f64,
}

/// Level proof statistics
#[derive(Debug, Serialize)]
pub struct LevelProofStats {
    pub level: i16,
    pub count: i64,
    pub avg_generation_time_ms: f64,
}

/// Performance statistics
#[derive(Debug, Serialize)]
pub struct PerformanceStats {
    pub avg_generation_time_ms: f64,
    pub median_generation_time_ms: f64,
    pub p99_generation_time_ms: f64,
    pub success_rate: f64,
}

/// Queue analytics
#[derive(Debug, Serialize)]
pub struct QueueAnalytics {
    pub current_length: usize,
    pub high_priority: usize,
    pub normal_priority: usize,
    pub low_priority: usize,
    pub active_generations: usize,
}

/// Get privacy analytics
pub async fn privacy_analytics(
    Extension(state): Extension<Arc<AppState>>,
    Query(query): Query<AnalyticsQuery>,
) -> Result<Json<PrivacyAnalytics>> {
    let end = query.end.unwrap_or_else(chrono::Utc::now);
    let start = query.start.unwrap_or_else(|| end - chrono::Duration::days(30));

    // Get morph stats
    let morph_stats: Option<(i64, i64, i64, Option<f64>, Option<f64>)> = sqlx::query_as(
        r#"
        SELECT 
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE new_level > previous_level) as upgrades,
            COUNT(*) FILTER (WHERE new_level < previous_level) as downgrades,
            AVG(previous_level) as avg_from,
            AVG(new_level) as avg_to
        FROM privacy_morphs
        WHERE morphed_at >= $1 AND morphed_at <= $2
        "#,
    )
    .bind(start)
    .bind(end)
    .fetch_optional(&state.db)
    .await
    .map_err(PrivacyError::Database)?;

    let morphs = match morph_stats {
        Some((total, upgrades, downgrades, avg_from, avg_to)) => MorphStats {
            total,
            upgrades,
            downgrades,
            average_from_level: avg_from.unwrap_or(0.0),
            average_to_level: avg_to.unwrap_or(0.0),
        },
        None => MorphStats {
            total: 0,
            upgrades: 0,
            downgrades: 0,
            average_from_level: 0.0,
            average_to_level: 0.0,
        },
    };

    // Get disclosure stats
    let disclosure_stats: Option<(i64, i64, i64)> = sqlx::query_as(
        r#"
        SELECT 
            COUNT(*) as total_created,
            COUNT(*) FILTER (WHERE revoked_at IS NOT NULL) as revoked,
            COUNT(*) FILTER (WHERE expires_at < NOW() AND revoked_at IS NULL) as expired
        FROM selective_disclosures
        WHERE created_at >= $1 AND created_at <= $2
        "#,
    )
    .bind(start)
    .bind(end)
    .fetch_optional(&state.db)
    .await
    .map_err(PrivacyError::Database)?;

    let disclosures = match disclosure_stats {
        Some((total_created, revoked, expired)) => DisclosureStats {
            total_created,
            total_verified: 0, // Would need to track verifications
            total_revoked: revoked,
            active: total_created - revoked - expired,
            expired,
        },
        None => DisclosureStats {
            total_created: 0,
            total_verified: 0,
            total_revoked: 0,
            active: 0,
            expired: 0,
        },
    };

    // Get level distribution
    let level_dist: Vec<(i16, i64)> = sqlx::query_as(
        r#"
        SELECT privacy_level, COUNT(*) as count
        FROM proof_jobs
        WHERE created_at >= $1 AND created_at <= $2
        GROUP BY privacy_level
        ORDER BY privacy_level
        "#,
    )
    .bind(start)
    .bind(end)
    .fetch_all(&state.db)
    .await
    .unwrap_or_default();

    let total_proofs: i64 = level_dist.iter().map(|(_, count)| count).sum();

    let level_distribution: Vec<LevelStats> = level_dist
        .iter()
        .map(|(level, count)| {
            let name = match level {
                0 => "Transparent",
                1 => "Sender-Shielded",
                2 => "Recipient-Shielded",
                3 => "Amount-Shielded",
                4 => "Full Privacy",
                5 => "Maximum",
                _ => "Unknown",
            };

            LevelStats {
                level: *level,
                name: name.to_string(),
                count: *count,
                percentage: if total_proofs > 0 {
                    (*count as f64 / total_proofs as f64) * 100.0
                } else {
                    0.0
                },
            }
        })
        .collect();

    Ok(Json(PrivacyAnalytics {
        period_start: start,
        period_end: end,
        morphs,
        disclosures,
        level_distribution,
    }))
}

/// Get proof analytics
pub async fn proof_analytics(
    Extension(state): Extension<Arc<AppState>>,
    Query(query): Query<AnalyticsQuery>,
) -> Result<Json<ProofAnalytics>> {
    let end = query.end.unwrap_or_else(chrono::Utc::now);
    let start = query.start.unwrap_or_else(|| end - chrono::Duration::days(30));

    // Get total and status breakdown
    let status_stats: Vec<(String, i64)> = sqlx::query_as(
        r#"
        SELECT status, COUNT(*) as count
        FROM proof_jobs
        WHERE created_at >= $1 AND created_at <= $2
        GROUP BY status
        "#,
    )
    .bind(start)
    .bind(end)
    .fetch_all(&state.db)
    .await
    .unwrap_or_default();

    let total_generated: i64 = status_stats.iter().map(|(_, count)| count).sum();

    let by_status: Vec<StatusStats> = status_stats
        .iter()
        .map(|(status, count)| StatusStats {
            status: status.clone(),
            count: *count,
            percentage: if total_generated > 0 {
                (*count as f64 / total_generated as f64) * 100.0
            } else {
                0.0
            },
        })
        .collect();

    // Get priority breakdown
    let priority_stats: Vec<(String, i64)> = sqlx::query_as(
        r#"
        SELECT priority, COUNT(*) as count
        FROM proof_jobs
        WHERE created_at >= $1 AND created_at <= $2
        GROUP BY priority
        "#,
    )
    .bind(start)
    .bind(end)
    .fetch_all(&state.db)
    .await
    .unwrap_or_default();

    let by_priority: Vec<PriorityStats> = priority_stats
        .iter()
        .map(|(priority, count)| PriorityStats {
            priority: priority.clone(),
            count: *count,
            avg_wait_ms: 0.0, // Would need to calculate
        })
        .collect();

    // Get level breakdown with avg generation time
    let level_stats: Vec<(i16, i64, Option<f64>)> = sqlx::query_as(
        r#"
        SELECT 
            privacy_level, 
            COUNT(*) as count,
            AVG(EXTRACT(EPOCH FROM (completed_at - created_at)) * 1000) as avg_time
        FROM proof_jobs
        WHERE created_at >= $1 AND created_at <= $2
        GROUP BY privacy_level
        ORDER BY privacy_level
        "#,
    )
    .bind(start)
    .bind(end)
    .fetch_all(&state.db)
    .await
    .unwrap_or_default();

    let by_level: Vec<LevelProofStats> = level_stats
        .iter()
        .map(|(level, count, avg_time)| LevelProofStats {
            level: *level,
            count: *count,
            avg_generation_time_ms: avg_time.unwrap_or(0.0),
        })
        .collect();

    // Calculate success rate
    let completed = status_stats
        .iter()
        .find(|(s, _)| s == "completed")
        .map(|(_, c)| *c)
        .unwrap_or(0);
    let success_rate = if total_generated > 0 {
        (completed as f64 / total_generated as f64) * 100.0
    } else {
        0.0
    };

    // Get current queue stats
    let queue_stats = state.get_queue_stats().await;

    Ok(Json(ProofAnalytics {
        period_start: start,
        period_end: end,
        total_generated,
        by_status,
        by_priority,
        by_level,
        performance: PerformanceStats {
            avg_generation_time_ms: level_stats
                .iter()
                .filter_map(|(_, _, avg)| *avg)
                .sum::<f64>()
                / level_stats.len().max(1) as f64,
            median_generation_time_ms: 0.0, // Would need full data
            p99_generation_time_ms: 0.0,
            success_rate,
        },
        queue: QueueAnalytics {
            current_length: queue_stats.total_queued,
            high_priority: queue_stats.high_priority,
            normal_priority: queue_stats.normal_priority,
            low_priority: queue_stats.low_priority,
            active_generations: queue_stats.active_generations,
        },
    }))
}
