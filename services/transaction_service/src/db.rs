//! Database operations for Transaction Service

use crate::error::{Result, TransactionError};
use crate::models::*;
use sqlx::PgPool;
use uuid::Uuid;

/// Transaction repository
pub struct TransactionRepository<'a> {
    db: &'a PgPool,
}

impl<'a> TransactionRepository<'a> {
    /// Create new repository instance
    pub fn new(db: &'a PgPool) -> Self {
        Self { db }
    }

    /// Create a new transaction
    pub async fn create(&self, user_id: Uuid, req: &CreateTransactionRequest) -> Result<Transaction> {
        let privacy_level = req.privacy_level.unwrap_or(4); // Default to FullPrivacy

        // Validate privacy level
        if privacy_level < 0 || privacy_level > 5 {
            return Err(TransactionError::InvalidPrivacyLevel(privacy_level));
        }

        let tx = sqlx::query_as::<_, Transaction>(
            r#"
            INSERT INTO transactions (
                id, user_id, sender, recipient, amount, asset_id,
                privacy_level, status, chain_id, memo, metadata,
                created_at, updated_at
            )
            VALUES (
                $1, $2, $3, $4, $5, $6, $7, 'pending', $8, $9, $10,
                NOW(), NOW()
            )
            RETURNING *
            "#,
        )
        .bind(Uuid::now_v7())
        .bind(user_id)
        .bind(&req.sender)
        .bind(&req.recipient)
        .bind(req.amount)
        .bind(&req.asset_id)
        .bind(privacy_level)
        .bind(&req.chain_id)
        .bind(&req.memo)
        .bind(&req.metadata)
        .fetch_one(self.db)
        .await?;

        Ok(tx)
    }

    /// Get transaction by ID
    pub async fn get_by_id(&self, id: Uuid) -> Result<Transaction> {
        sqlx::query_as::<_, Transaction>("SELECT * FROM transactions WHERE id = $1")
            .bind(id)
            .fetch_optional(self.db)
            .await?
            .ok_or_else(|| TransactionError::NotFound(id.to_string()))
    }

    /// Get transaction by ID for a specific user
    pub async fn get_by_id_for_user(&self, id: Uuid, user_id: Uuid) -> Result<Transaction> {
        sqlx::query_as::<_, Transaction>(
            "SELECT * FROM transactions WHERE id = $1 AND user_id = $2",
        )
        .bind(id)
        .bind(user_id)
        .fetch_optional(self.db)
        .await?
        .ok_or_else(|| TransactionError::NotFound(id.to_string()))
    }

    /// List transactions with filtering and pagination
    pub async fn list(
        &self,
        user_id: Uuid,
        query: &ListTransactionsQuery,
    ) -> Result<TransactionListResponse> {
        let offset = (query.page.saturating_sub(1)) * query.page_size;

        // Build dynamic query
        let mut sql = String::from(
            r#"
            SELECT * FROM transactions
            WHERE user_id = $1
            "#,
        );

        let mut param_idx = 2;
        let mut bindings: Vec<String> = vec![];

        if let Some(status) = &query.status {
            sql.push_str(&format!(" AND status = ${}", param_idx));
            bindings.push(format!("{:?}", status).to_lowercase());
            param_idx += 1;
        }

        if let Some(level) = query.privacy_level {
            sql.push_str(&format!(" AND privacy_level = ${}", param_idx));
            bindings.push(level.to_string());
            param_idx += 1;
        }

        if let Some(chain) = &query.chain_id {
            sql.push_str(&format!(" AND chain_id = ${}", param_idx));
            bindings.push(chain.clone());
            param_idx += 1;
        }

        if let Some(asset) = &query.asset_id {
            sql.push_str(&format!(" AND asset_id = ${}", param_idx));
            bindings.push(asset.clone());
            param_idx += 1;
        }

        // Add ordering
        let order_dir = if query.sort_desc { "DESC" } else { "ASC" };
        let sort_column = match query.sort_by.as_str() {
            "created_at" | "updated_at" | "amount" | "status" | "privacy_level" => {
                query.sort_by.clone()
            }
            _ => "created_at".to_string(),
        };
        sql.push_str(&format!(" ORDER BY {} {}", sort_column, order_dir));

        // Add pagination
        sql.push_str(&format!(" LIMIT {} OFFSET {}", query.page_size, offset));

        // For now, use simple query without dynamic parameters
        let transactions = sqlx::query_as::<_, Transaction>(
            r#"
            SELECT * FROM transactions
            WHERE user_id = $1
            ORDER BY created_at DESC
            LIMIT $2 OFFSET $3
            "#,
        )
        .bind(user_id)
        .bind(query.page_size as i64)
        .bind(offset as i64)
        .fetch_all(self.db)
        .await?;

        // Get total count
        let total: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM transactions WHERE user_id = $1")
            .bind(user_id)
            .fetch_one(self.db)
            .await?;

        let total_pages = (total.0 as u32 + query.page_size - 1) / query.page_size;

        let summaries: Vec<TransactionSummary> = transactions
            .iter()
            .map(|tx| TransactionSummary {
                id: tx.id,
                status: tx.status,
                privacy_level: tx.privacy_level,
                amount: if tx.privacy_level < 3 {
                    Some(tx.amount)
                } else {
                    None
                },
                asset_id: tx.asset_id.clone(),
                chain_id: tx.chain_id.clone(),
                has_proof: tx.proof.is_some(),
                created_at: tx.created_at,
            })
            .collect();

        Ok(TransactionListResponse {
            transactions: summaries,
            total: total.0,
            page: query.page,
            page_size: query.page_size,
            total_pages,
        })
    }

    /// Update transaction
    pub async fn update(
        &self,
        id: Uuid,
        user_id: Uuid,
        req: &UpdateTransactionRequest,
    ) -> Result<Transaction> {
        // First check if transaction exists and belongs to user
        let tx = self.get_by_id_for_user(id, user_id).await?;

        // Only allow updates in certain states
        if !matches!(tx.status, TransactionStatus::Pending | TransactionStatus::ProofReady) {
            return Err(TransactionError::InvalidRequest(
                "Transaction cannot be updated in current state".to_string(),
            ));
        }

        let updated = sqlx::query_as::<_, Transaction>(
            r#"
            UPDATE transactions
            SET memo = COALESCE($3, memo),
                metadata = COALESCE($4, metadata),
                updated_at = NOW()
            WHERE id = $1 AND user_id = $2
            RETURNING *
            "#,
        )
        .bind(id)
        .bind(user_id)
        .bind(&req.memo)
        .bind(&req.metadata)
        .fetch_one(self.db)
        .await?;

        Ok(updated)
    }

    /// Update transaction status
    pub async fn update_status(
        &self,
        id: Uuid,
        new_status: TransactionStatus,
    ) -> Result<Transaction> {
        let tx = self.get_by_id(id).await?;

        // Validate status transition
        if !Self::is_valid_transition(tx.status, new_status) {
            return Err(TransactionError::InvalidStatusTransition {
                from: tx.status,
                to: new_status,
            });
        }

        let updated = sqlx::query_as::<_, Transaction>(
            r#"
            UPDATE transactions
            SET status = $2,
                updated_at = NOW(),
                finalized_at = CASE WHEN $2 = 'finalized' THEN NOW() ELSE finalized_at END
            WHERE id = $1
            RETURNING *
            "#,
        )
        .bind(id)
        .bind(new_status)
        .fetch_one(self.db)
        .await?;

        Ok(updated)
    }

    /// Check if status transition is valid
    fn is_valid_transition(from: TransactionStatus, to: TransactionStatus) -> bool {
        use TransactionStatus::*;

        matches!(
            (from, to),
            (Pending, ProofGenerating)
                | (Pending, Cancelled)
                | (ProofGenerating, ProofReady)
                | (ProofGenerating, Failed)
                | (ProofReady, Submitted)
                | (ProofReady, Cancelled)
                | (Submitted, Confirmed)
                | (Submitted, Failed)
                | (Confirmed, Finalized)
                | (Confirmed, Failed)
        )
    }

    /// Update privacy level
    pub async fn update_privacy_level(
        &self,
        id: Uuid,
        new_level: i16,
    ) -> Result<Transaction> {
        if new_level < 0 || new_level > 5 {
            return Err(TransactionError::InvalidPrivacyLevel(new_level));
        }

        let tx = self.get_by_id(id).await?;

        // Only allow privacy morphing in pending or proof_ready states
        if !matches!(tx.status, TransactionStatus::Pending | TransactionStatus::ProofReady) {
            return Err(TransactionError::PrivacyMorphFailed(
                "Cannot change privacy level after submission".to_string(),
            ));
        }

        // Invalidate existing proof if privacy level changes
        let needs_new_proof = tx.privacy_level != new_level && tx.proof.is_some();

        let updated = if needs_new_proof {
            sqlx::query_as::<_, Transaction>(
                r#"
                UPDATE transactions
                SET privacy_level = $2,
                    proof = NULL,
                    proof_id = NULL,
                    status = 'pending',
                    updated_at = NOW()
                WHERE id = $1
                RETURNING *
                "#,
            )
            .bind(id)
            .bind(new_level)
            .fetch_one(self.db)
            .await?
        } else {
            sqlx::query_as::<_, Transaction>(
                r#"
                UPDATE transactions
                SET privacy_level = $2,
                    updated_at = NOW()
                WHERE id = $1
                RETURNING *
                "#,
            )
            .bind(id)
            .bind(new_level)
            .fetch_one(self.db)
            .await?
        };

        Ok(updated)
    }

    /// Store proof for transaction
    pub async fn store_proof(
        &self,
        id: Uuid,
        proof_id: Uuid,
        proof: &str,
    ) -> Result<Transaction> {
        let updated = sqlx::query_as::<_, Transaction>(
            r#"
            UPDATE transactions
            SET proof = $2,
                proof_id = $3,
                status = 'proof_ready',
                updated_at = NOW()
            WHERE id = $1
            RETURNING *
            "#,
        )
        .bind(id)
        .bind(proof)
        .bind(proof_id)
        .fetch_one(self.db)
        .await?;

        Ok(updated)
    }

    /// Cancel transaction
    pub async fn cancel(&self, id: Uuid, user_id: Uuid) -> Result<Transaction> {
        let tx = self.get_by_id_for_user(id, user_id).await?;

        if !matches!(
            tx.status,
            TransactionStatus::Pending | TransactionStatus::ProofReady
        ) {
            return Err(TransactionError::InvalidRequest(
                "Transaction cannot be cancelled in current state".to_string(),
            ));
        }

        let updated = sqlx::query_as::<_, Transaction>(
            r#"
            UPDATE transactions
            SET status = 'cancelled',
                updated_at = NOW()
            WHERE id = $1 AND user_id = $2
            RETURNING *
            "#,
        )
        .bind(id)
        .bind(user_id)
        .fetch_one(self.db)
        .await?;

        Ok(updated)
    }

    /// Get analytics summary
    pub async fn get_analytics_summary(
        &self,
        user_id: Uuid,
        start: chrono::DateTime<chrono::Utc>,
        end: chrono::DateTime<chrono::Utc>,
    ) -> Result<AnalyticsSummary> {
        // Total transactions
        let total: (i64,) = sqlx::query_as(
            r#"
            SELECT COUNT(*) FROM transactions
            WHERE user_id = $1 AND created_at >= $2 AND created_at <= $3
            "#,
        )
        .bind(user_id)
        .bind(start)
        .bind(end)
        .fetch_one(self.db)
        .await?;

        // By status
        let by_status: Vec<(String, i64)> = sqlx::query_as(
            r#"
            SELECT status::text, COUNT(*) FROM transactions
            WHERE user_id = $1 AND created_at >= $2 AND created_at <= $3
            GROUP BY status
            "#,
        )
        .bind(user_id)
        .bind(start)
        .bind(end)
        .fetch_all(self.db)
        .await?;

        // By privacy level
        let by_privacy: Vec<(i16, i64)> = sqlx::query_as(
            r#"
            SELECT privacy_level, COUNT(*) FROM transactions
            WHERE user_id = $1 AND created_at >= $2 AND created_at <= $3
            GROUP BY privacy_level
            "#,
        )
        .bind(user_id)
        .bind(start)
        .bind(end)
        .fetch_all(self.db)
        .await?;

        // Total volume
        let volume: (Option<i64>,) = sqlx::query_as(
            r#"
            SELECT SUM(amount) FROM transactions
            WHERE user_id = $1 AND created_at >= $2 AND created_at <= $3
            AND status IN ('confirmed', 'finalized')
            "#,
        )
        .bind(user_id)
        .bind(start)
        .bind(end)
        .fetch_one(self.db)
        .await?;

        // Success rate
        let success_count: (i64,) = sqlx::query_as(
            r#"
            SELECT COUNT(*) FROM transactions
            WHERE user_id = $1 AND created_at >= $2 AND created_at <= $3
            AND status IN ('confirmed', 'finalized')
            "#,
        )
        .bind(user_id)
        .bind(start)
        .bind(end)
        .fetch_one(self.db)
        .await?;

        let success_rate = if total.0 > 0 {
            (success_count.0 as f64 / total.0 as f64) * 100.0
        } else {
            0.0
        };

        Ok(AnalyticsSummary {
            total_transactions: total.0,
            by_status: by_status.into_iter().collect(),
            by_privacy_level: by_privacy.into_iter().collect(),
            total_volume: volume.0.unwrap_or(0),
            avg_proof_time_ms: 0.0, // TODO: Track proof generation time
            success_rate,
            period_start: start,
            period_end: end,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_transitions() {
        use TransactionStatus::*;

        assert!(TransactionRepository::is_valid_transition(Pending, ProofGenerating));
        assert!(TransactionRepository::is_valid_transition(Pending, Cancelled));
        assert!(TransactionRepository::is_valid_transition(Confirmed, Finalized));

        assert!(!TransactionRepository::is_valid_transition(Pending, Finalized));
        assert!(!TransactionRepository::is_valid_transition(Cancelled, Pending));
        assert!(!TransactionRepository::is_valid_transition(Finalized, Pending));
    }
}
