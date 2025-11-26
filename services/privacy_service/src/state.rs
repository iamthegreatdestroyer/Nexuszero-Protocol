//! Application state for Privacy Service

use crate::config::Config;
use crate::models::{ProofGenerationRequest, ProofPriority, ProofStatus};
use sqlx::postgres::PgPoolOptions;
use sqlx::PgPool;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Mutex, RwLock, Semaphore};
use uuid::Uuid;

/// Shared application state
pub struct AppState {
    /// PostgreSQL connection pool
    pub db: PgPool,

    /// Redis client
    pub redis: redis::Client,

    /// Configuration
    pub config: Config,

    /// HTTP client
    pub http_client: reqwest::Client,

    /// Proof generation queue
    pub proof_queue: Arc<Mutex<ProofQueue>>,

    /// Proof generation semaphore (limits concurrent generation)
    pub proof_semaphore: Arc<Semaphore>,

    /// Active proof generations
    pub active_proofs: Arc<RwLock<HashMap<Uuid, ProofJob>>>,
}

/// Proof generation queue
pub struct ProofQueue {
    /// High priority queue
    pub high: Vec<ProofJob>,

    /// Normal priority queue
    pub normal: Vec<ProofJob>,

    /// Low priority queue
    pub low: Vec<ProofJob>,
}

impl ProofQueue {
    pub fn new() -> Self {
        Self {
            high: Vec::new(),
            normal: Vec::new(),
            low: Vec::new(),
        }
    }

    pub fn enqueue(&mut self, job: ProofJob) {
        match job.priority {
            ProofPriority::High => self.high.push(job),
            ProofPriority::Normal => self.normal.push(job),
            ProofPriority::Low => self.low.push(job),
        }
    }

    pub fn dequeue(&mut self) -> Option<ProofJob> {
        if let Some(job) = self.high.pop() {
            return Some(job);
        }
        if let Some(job) = self.normal.pop() {
            return Some(job);
        }
        self.low.pop()
    }

    pub fn len(&self) -> usize {
        self.high.len() + self.normal.len() + self.low.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn position(&self, proof_id: Uuid) -> Option<u32> {
        // Check high priority first
        for (i, job) in self.high.iter().enumerate() {
            if job.proof_id == proof_id {
                return Some(i as u32);
            }
        }

        // Then normal
        let high_len = self.high.len() as u32;
        for (i, job) in self.normal.iter().enumerate() {
            if job.proof_id == proof_id {
                return Some(high_len + i as u32);
            }
        }

        // Then low
        let normal_len = self.normal.len() as u32;
        for (i, job) in self.low.iter().enumerate() {
            if job.proof_id == proof_id {
                return Some(high_len + normal_len + i as u32);
            }
        }

        None
    }
}

impl Default for ProofQueue {
    fn default() -> Self {
        Self::new()
    }
}

/// Proof generation job
#[derive(Debug, Clone)]
pub struct ProofJob {
    pub proof_id: Uuid,
    pub transaction_id: Uuid,
    pub sender: String,
    pub recipient: String,
    pub amount: i64,
    pub privacy_level: i16,
    pub priority: ProofPriority,
    pub callback_url: Option<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub status: ProofStatus,
}

impl ProofJob {
    pub fn from_request(request: &ProofGenerationRequest) -> Self {
        Self {
            proof_id: Uuid::now_v7(),
            transaction_id: request.transaction_id,
            sender: request.sender.clone(),
            recipient: request.recipient.clone(),
            amount: request.amount,
            privacy_level: request.privacy_level,
            priority: request.priority,
            callback_url: request.callback_url.clone(),
            created_at: chrono::Utc::now(),
            status: ProofStatus::Queued,
        }
    }
}

impl AppState {
    /// Create new application state
    pub async fn new(config: &Config) -> anyhow::Result<Self> {
        tracing::info!("Initializing Privacy Service state...");

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
            .query_async::<_, String>(&mut conn)
            .await?;
        tracing::info!("Redis connection established");

        // Initialize HTTP client
        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .connect_timeout(Duration::from_secs(10))
            .build()?;

        // Initialize proof queue
        let proof_queue = Arc::new(Mutex::new(ProofQueue::new()));

        // Initialize proof semaphore
        let proof_semaphore = Arc::new(Semaphore::new(config.proof.max_concurrent));

        // Initialize active proofs tracking
        let active_proofs = Arc::new(RwLock::new(HashMap::new()));

        let state = Self {
            db,
            redis,
            config: config.clone(),
            http_client,
            proof_queue,
            proof_semaphore,
            active_proofs,
        };

        // Start proof processor
        state.start_proof_processor();

        tracing::info!("Privacy Service state initialized");

        Ok(state)
    }

    /// Start the background proof processor
    fn start_proof_processor(&self) {
        let queue = self.proof_queue.clone();
        let semaphore = self.proof_semaphore.clone();
        let active = self.active_proofs.clone();
        let config = self.config.clone();
        let db = self.db.clone();
        let http_client = self.http_client.clone();

        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_millis(100)).await;

                // Try to get a permit
                let permit = match semaphore.clone().try_acquire_owned() {
                    Ok(p) => p,
                    Err(_) => continue, // All workers busy
                };

                // Get next job from queue
                let job = {
                    let mut q = queue.lock().await;
                    q.dequeue()
                };

                if let Some(mut job) = job {
                    job.status = ProofStatus::Generating;

                    // Track active job
                    {
                        let mut active_map = active.write().await;
                        active_map.insert(job.proof_id, job.clone());
                    }

                    let active_clone = active.clone();
                    let db_clone = db.clone();
                    let http_clone = http_client.clone();
                    let timeout = config.proof.timeout_secs;

                    tokio::spawn(async move {
                        let result = tokio::time::timeout(
                            Duration::from_secs(timeout),
                            Self::generate_proof_for_job(&job),
                        )
                        .await;

                        let (status, proof, error) = match result {
                            Ok(Ok(proof_data)) => (ProofStatus::Completed, Some(proof_data), None),
                            Ok(Err(e)) => (ProofStatus::Failed, None, Some(e.to_string())),
                            Err(_) => (ProofStatus::Failed, None, Some("Timeout".to_string())),
                        };

                        // Update database
                        let _ = sqlx::query(
                            r#"
                            UPDATE proof_jobs 
                            SET status = $2, proof = $3, error = $4, completed_at = NOW()
                            WHERE id = $1
                            "#,
                        )
                        .bind(job.proof_id)
                        .bind(format!("{:?}", status).to_lowercase())
                        .bind(&proof)
                        .bind(&error)
                        .execute(&db_clone)
                        .await;

                        // Callback if provided
                        if let Some(url) = &job.callback_url {
                            let _ = http_clone
                                .post(url)
                                .json(&serde_json::json!({
                                    "proof_id": job.proof_id,
                                    "transaction_id": job.transaction_id,
                                    "status": format!("{:?}", status).to_lowercase(),
                                    "proof": proof,
                                    "error": error,
                                }))
                                .send()
                                .await;
                        }

                        // Remove from active
                        {
                            let mut active_map = active_clone.write().await;
                            active_map.remove(&job.proof_id);
                        }

                        drop(permit);
                    });
                }
            }
        });
    }

    /// Generate proof for a job (placeholder - would use actual ZK proving)
    async fn generate_proof_for_job(job: &ProofJob) -> anyhow::Result<String> {
        use sha2::{Sha256, Digest};

        // Simulate proof generation time based on privacy level
        let delay_ms = match job.privacy_level {
            0 => 0,
            1 | 2 => 500,
            3 => 1000,
            4 => 2500,
            5 => 5000,
            _ => 1500,
        };

        tokio::time::sleep(Duration::from_millis(delay_ms)).await;

        // Generate a placeholder proof
        let mut hasher = Sha256::new();
        hasher.update(job.transaction_id.as_bytes());
        hasher.update(job.sender.as_bytes());
        hasher.update(job.recipient.as_bytes());
        hasher.update(job.amount.to_le_bytes());
        hasher.update(job.privacy_level.to_le_bytes());
        hasher.update(chrono::Utc::now().timestamp().to_le_bytes());

        let hash = hasher.finalize();
        let proof = format!("proof_v1_{}", hex::encode(hash));

        Ok(proof)
    }

    /// Get a Redis connection
    pub async fn redis_conn(&self) -> anyhow::Result<redis::aio::MultiplexedConnection> {
        Ok(self.redis.get_multiplexed_async_connection().await?)
    }

    /// Check database health
    pub async fn check_db_health(&self) -> bool {
        sqlx::query("SELECT 1").execute(&self.db).await.is_ok()
    }

    /// Check Redis health
    pub async fn check_redis_health(&self) -> bool {
        if let Ok(mut conn) = self.redis.get_multiplexed_async_connection().await {
            redis::cmd("PING")
                .query_async::<_, String>(&mut conn)
                .await
                .is_ok()
        } else {
            false
        }
    }

    /// Get queue statistics
    pub async fn get_queue_stats(&self) -> QueueStats {
        let queue = self.proof_queue.lock().await;
        let active = self.active_proofs.read().await;

        QueueStats {
            high_priority: queue.high.len(),
            normal_priority: queue.normal.len(),
            low_priority: queue.low.len(),
            total_queued: queue.len(),
            active_generations: active.len(),
            max_concurrent: self.config.proof.max_concurrent,
        }
    }
}

/// Queue statistics
#[derive(Debug, Clone, serde::Serialize)]
pub struct QueueStats {
    pub high_priority: usize,
    pub normal_priority: usize,
    pub low_priority: usize,
    pub total_queued: usize,
    pub active_generations: usize,
    pub max_concurrent: usize,
}
