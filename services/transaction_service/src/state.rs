//! Application state for Transaction Service

use crate::config::Config;
use sqlx::postgres::PgPoolOptions;
use sqlx::PgPool;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

/// Shared application state
pub struct AppState {
    /// PostgreSQL connection pool
    pub db: PgPool,

    /// Redis client
    pub redis: redis::Client,

    /// Configuration
    pub config: Config,

    /// HTTP client for service-to-service communication
    pub http_client: reqwest::Client,

    /// Proof generation queue (in-memory for now)
    pub proof_queue: Arc<RwLock<ProofQueue>>,
}

/// Simple in-memory proof queue
pub struct ProofQueue {
    pub pending: Vec<uuid::Uuid>,
    pub processing: std::collections::HashSet<uuid::Uuid>,
}

impl ProofQueue {
    pub fn new() -> Self {
        Self {
            pending: Vec::new(),
            processing: std::collections::HashSet::new(),
        }
    }

    pub fn enqueue(&mut self, transaction_id: uuid::Uuid) {
        if !self.processing.contains(&transaction_id) {
            self.pending.push(transaction_id);
        }
    }

    pub fn dequeue(&mut self) -> Option<uuid::Uuid> {
        if let Some(id) = self.pending.pop() {
            self.processing.insert(id);
            Some(id)
        } else {
            None
        }
    }

    pub fn complete(&mut self, transaction_id: &uuid::Uuid) {
        self.processing.remove(transaction_id);
    }

    pub fn queue_length(&self) -> usize {
        self.pending.len()
    }

    pub fn processing_count(&self) -> usize {
        self.processing.len()
    }
}

impl Default for ProofQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl AppState {
    /// Create new application state
    pub async fn new(config: &Config) -> anyhow::Result<Self> {
        tracing::info!("Initializing Transaction Service state...");

        // Initialize PostgreSQL connection pool
        tracing::info!("Connecting to PostgreSQL...");
        let db = PgPoolOptions::new()
            .max_connections(config.database.max_connections)
            .min_connections(config.database.min_connections)
            .acquire_timeout(Duration::from_secs(config.database.connect_timeout_secs))
            .connect(&config.database.url)
            .await?;

        // Run migrations
        sqlx::migrate!("./migrations").run(&db).await?;
        tracing::info!("Database migrations completed");

        // Initialize Redis client
        tracing::info!("Connecting to Redis...");
        let redis = redis::Client::open(config.redis.url.clone())?;

        // Test Redis connection
        let mut conn = redis.get_multiplexed_async_connection().await?;
        redis::cmd("PING")
            .query_async::<String>(&mut conn)
            .await?;
        tracing::info!("Redis connection established");

        // Initialize HTTP client
        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .connect_timeout(Duration::from_secs(10))
            .pool_max_idle_per_host(10)
            .build()?;

        // Initialize proof queue
        let proof_queue = Arc::new(RwLock::new(ProofQueue::new()));

        // Start proof processor task
        let proof_queue_clone = proof_queue.clone();
        let db_clone = db.clone();
        let config_clone = config.clone();
        let http_client_clone = http_client.clone();

        tokio::spawn(async move {
            Self::proof_processor(proof_queue_clone, db_clone, config_clone, http_client_clone)
                .await;
        });

        tracing::info!("Transaction Service state initialized");

        Ok(Self {
            db,
            redis,
            config: config.clone(),
            http_client,
            proof_queue,
        })
    }

    /// Get a Redis connection
    pub async fn redis_conn(&self) -> anyhow::Result<redis::aio::MultiplexedConnection> {
        Ok(self.redis.get_multiplexed_async_connection().await?)
    }

    /// Background task to process proof generation
    async fn proof_processor(
        queue: Arc<RwLock<ProofQueue>>,
        db: PgPool,
        config: Config,
        http_client: reqwest::Client,
    ) {
        let interval = Duration::from_secs(5);

        loop {
            tokio::time::sleep(interval).await;

            // Get next transaction to process
            let transaction_id = {
                let mut q = queue.write().await;
                q.dequeue()
            };

            if let Some(tx_id) = transaction_id {
                tracing::debug!(transaction_id = %tx_id, "Processing proof generation");

                match Self::generate_proof_for_transaction(&db, &config, &http_client, tx_id).await
                {
                    Ok(()) => {
                        tracing::info!(transaction_id = %tx_id, "Proof generated successfully");
                    }
                    Err(e) => {
                        tracing::error!(
                            transaction_id = %tx_id,
                            error = %e,
                            "Proof generation failed"
                        );
                    }
                }

                // Mark as complete
                let mut q = queue.write().await;
                q.complete(&tx_id);
            }
        }
    }

    /// Generate proof for a transaction
    async fn generate_proof_for_transaction(
        db: &PgPool,
        config: &Config,
        http_client: &reqwest::Client,
        transaction_id: uuid::Uuid,
    ) -> anyhow::Result<()> {
        // Update status to ProofGenerating
        sqlx::query(
            "UPDATE transactions SET status = 'proof_generating', updated_at = NOW() WHERE id = $1",
        )
        .bind(transaction_id)
        .execute(db)
        .await?;

        // Fetch transaction details
        let tx: (String, String, i64, i16) = sqlx::query_as(
            "SELECT sender, recipient, amount, privacy_level FROM transactions WHERE id = $1",
        )
        .bind(transaction_id)
        .fetch_one(db)
        .await?;

        // Call Privacy Service to generate proof
        let proof_response = http_client
            .post(format!(
                "{}/api/v1/proof/generate",
                config.privacy_service_url
            ))
            .json(&serde_json::json!({
                "transaction_id": transaction_id,
                "sender": tx.0,
                "recipient": tx.1,
                "amount": tx.2,
                "privacy_level": tx.3,
            }))
            .send()
            .await?;

        if proof_response.status().is_success() {
            let proof_data: serde_json::Value = proof_response.json().await?;

            // Update transaction with proof
            sqlx::query(
                r#"
                UPDATE transactions 
                SET status = 'proof_ready',
                    proof = $2,
                    proof_id = $3,
                    updated_at = NOW()
                WHERE id = $1
                "#,
            )
            .bind(transaction_id)
            .bind(proof_data["proof"].as_str())
            .bind(
                proof_data["proof_id"]
                    .as_str()
                    .and_then(|s| uuid::Uuid::parse_str(s).ok()),
            )
            .execute(db)
            .await?;
        } else {
            let error_msg = proof_response.text().await.unwrap_or_default();

            sqlx::query(
                r#"
                UPDATE transactions 
                SET status = 'failed',
                    error_message = $2,
                    updated_at = NOW()
                WHERE id = $1
                "#,
            )
            .bind(transaction_id)
            .bind(error_msg)
            .execute(db)
            .await?;
        }

        Ok(())
    }

    /// Check database health
    pub async fn check_db_health(&self) -> bool {
        sqlx::query("SELECT 1").execute(&self.db).await.is_ok()
    }

    /// Check Redis health
    pub async fn check_redis_health(&self) -> bool {
        if let Ok(mut conn) = self.redis.get_multiplexed_async_connection().await {
            redis::cmd("PING")
                .query_async::<String>(&mut conn)
                .await
                .is_ok()
        } else {
            false
        }
    }

    /// Check Privacy Service health
    pub async fn check_privacy_service_health(&self) -> bool {
        self.http_client
            .get(format!("{}/health", self.config.privacy_service_url))
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }

    /// Check Compliance Service health
    pub async fn check_compliance_service_health(&self) -> bool {
        self.http_client
            .get(format!("{}/health", self.config.compliance_service_url))
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }
}
