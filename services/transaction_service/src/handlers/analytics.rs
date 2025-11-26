//! Analytics handlers

use crate::db::TransactionRepository;
use crate::error::Result;
use crate::models::*;
use crate::state::AppState;
use axum::{
    extract::{Extension, Query},
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

/// Temporary user ID extraction
fn get_user_id() -> Uuid {
    Uuid::parse_str("00000000-0000-0000-0000-000000000001").unwrap()
}

/// Analytics query parameters
#[derive(Debug, Deserialize)]
pub struct AnalyticsQuery {
    /// Period start (ISO 8601)
    pub start: Option<chrono::DateTime<chrono::Utc>>,

    /// Period end (ISO 8601)
    pub end: Option<chrono::DateTime<chrono::Utc>>,

    /// Chain filter
    pub chain_id: Option<String>,

    /// Asset filter
    pub asset_id: Option<String>,
}

/// Privacy distribution response
#[derive(Debug, Serialize)]
pub struct PrivacyDistribution {
    pub period_start: chrono::DateTime<chrono::Utc>,
    pub period_end: chrono::DateTime<chrono::Utc>,
    pub total_transactions: i64,
    pub distribution: Vec<PrivacyLevelStats>,
    pub average_privacy_level: f64,
    pub trends: PrivacyTrends,
}

/// Stats for a single privacy level
#[derive(Debug, Serialize)]
pub struct PrivacyLevelStats {
    pub level: i16,
    pub name: String,
    pub count: i64,
    pub percentage: f64,
    pub volume: i64,
}

/// Privacy trends
#[derive(Debug, Serialize)]
pub struct PrivacyTrends {
    pub privacy_increasing: bool,
    pub most_popular_level: i16,
    pub highest_volume_level: i16,
}

/// Get analytics summary
pub async fn get_summary(
    Extension(state): Extension<Arc<AppState>>,
    Query(query): Query<AnalyticsQuery>,
) -> Result<Json<AnalyticsSummary>> {
    let user_id = get_user_id();
    let repo = TransactionRepository::new(&state.db);

    // Default to last 30 days
    let end = query.end.unwrap_or_else(chrono::Utc::now);
    let start = query.start.unwrap_or_else(|| end - chrono::Duration::days(30));

    let summary = repo.get_analytics_summary(user_id, start, end).await?;

    Ok(Json(summary))
}

/// Get privacy level distribution
pub async fn privacy_distribution(
    Extension(state): Extension<Arc<AppState>>,
    Query(query): Query<AnalyticsQuery>,
) -> Result<Json<PrivacyDistribution>> {
    let user_id = get_user_id();

    // Default to last 30 days
    let end = query.end.unwrap_or_else(chrono::Utc::now);
    let start = query.start.unwrap_or_else(|| end - chrono::Duration::days(30));

    // Get distribution from database
    let distribution: Vec<(i16, i64, Option<i64>)> = sqlx::query_as(
        r#"
        SELECT 
            privacy_level,
            COUNT(*) as count,
            SUM(amount) as volume
        FROM transactions
        WHERE user_id = $1 AND created_at >= $2 AND created_at <= $3
        GROUP BY privacy_level
        ORDER BY privacy_level
        "#,
    )
    .bind(user_id)
    .bind(start)
    .bind(end)
    .fetch_all(&state.db)
    .await
    .unwrap_or_default();

    let total: i64 = distribution.iter().map(|(_, count, _)| count).sum();

    let level_stats: Vec<PrivacyLevelStats> = distribution
        .iter()
        .map(|(level, count, volume)| {
            let name = match level {
                0 => "Transparent",
                1 => "Sender-Shielded",
                2 => "Recipient-Shielded",
                3 => "Amount-Shielded",
                4 => "Full Privacy",
                5 => "Maximum",
                _ => "Unknown",
            };

            PrivacyLevelStats {
                level: *level,
                name: name.to_string(),
                count: *count,
                percentage: if total > 0 {
                    (*count as f64 / total as f64) * 100.0
                } else {
                    0.0
                },
                volume: volume.unwrap_or(0),
            }
        })
        .collect();

    // Calculate average privacy level
    let weighted_sum: f64 = distribution
        .iter()
        .map(|(level, count, _)| *level as f64 * *count as f64)
        .sum();
    let average_privacy_level = if total > 0 {
        weighted_sum / total as f64
    } else {
        0.0
    };

    // Find most popular and highest volume levels
    let most_popular = distribution
        .iter()
        .max_by_key(|(_, count, _)| count)
        .map(|(level, _, _)| *level)
        .unwrap_or(4);

    let highest_volume = distribution
        .iter()
        .max_by_key(|(_, _, volume)| volume.unwrap_or(0))
        .map(|(level, _, _)| *level)
        .unwrap_or(4);

    // Determine if privacy is trending up
    // This would require historical data comparison in a real implementation
    let privacy_increasing = average_privacy_level >= 3.5;

    Ok(Json(PrivacyDistribution {
        period_start: start,
        period_end: end,
        total_transactions: total,
        distribution: level_stats,
        average_privacy_level,
        trends: PrivacyTrends {
            privacy_increasing,
            most_popular_level: most_popular,
            highest_volume_level: highest_volume,
        },
    }))
}
