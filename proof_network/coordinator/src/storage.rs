//! Storage Backend for Coordinator
//!
//! Provides storage abstraction for tasks, provers, and proof results.
//! Includes an in-memory implementation for development and testing.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use uuid::Uuid;

/// Task status in the coordinator
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskStatus {
    /// Task is pending assignment
    Pending,
    /// Task has been assigned to a prover
    Assigned { prover_id: Uuid },
    /// Task is currently being processed
    InProgress { prover_id: Uuid, started_at: DateTime<Utc> },
    /// Task has been completed
    Completed { prover_id: Uuid, completed_at: DateTime<Utc> },
    /// Task has failed
    Failed { reason: String },
}

/// Stored task with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredTask {
    /// Unique task identifier
    pub task_id: Uuid,
    /// Privacy level required (1-5)
    pub privacy_level: u8,
    /// Circuit data for proof generation
    pub circuit_data: Vec<u8>,
    /// Reward amount in tokens
    pub reward_amount: u64,
    /// Task requester identifier
    pub requester: String,
    /// Current status
    pub status: TaskStatus,
    /// Priority (higher = more important)
    pub priority: u32,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last update timestamp
    pub updated_at: DateTime<Utc>,
}

/// Stored prover with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredProver {
    /// Unique prover identifier
    pub prover_id: Uuid,
    /// Supported privacy levels
    pub supported_levels: Vec<u8>,
    /// Whether prover is currently active
    pub is_active: bool,
    /// Last heartbeat timestamp
    pub last_seen: DateTime<Utc>,
    /// Registration timestamp
    pub registered_at: DateTime<Utc>,
    /// Current capacity (number of concurrent tasks)
    pub capacity: u32,
}

/// Stored proof result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredResult {
    /// Result identifier
    pub result_id: Uuid,
    /// Associated task identifier
    pub task_id: Uuid,
    /// Prover who generated the proof
    pub prover_id: Uuid,
    /// The generated proof
    pub proof: Vec<u8>,
    /// Submission timestamp
    pub submitted_at: DateTime<Utc>,
    /// Verification status
    pub verified: Option<bool>,
}

/// Storage error types
#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    #[error("Task not found: {0}")]
    TaskNotFound(Uuid),
    #[error("Prover not found: {0}")]
    ProverNotFound(Uuid),
    #[error("Result not found: {0}")]
    ResultNotFound(Uuid),
    #[error("Storage error: {0}")]
    Internal(String),
}

/// Result type for storage operations
pub type StorageResult<T> = Result<T, StorageError>;

/// Async storage backend trait for coordinator data
#[async_trait]
pub trait StorageBackend: Send + Sync {
    /// Save a task to storage
    async fn save_task(&self, task: StoredTask) -> StorageResult<()>;

    /// Get a task by ID
    async fn get_task(&self, task_id: Uuid) -> StorageResult<StoredTask>;

    /// List all pending tasks, sorted by priority (descending)
    async fn list_pending_tasks(&self) -> StorageResult<Vec<StoredTask>>;

    /// Save a prover to storage
    async fn save_prover(&self, prover: StoredProver) -> StorageResult<()>;

    /// Get a prover by ID
    async fn get_prover(&self, prover_id: Uuid) -> StorageResult<StoredProver>;

    /// List all active provers
    async fn list_active_provers(&self) -> StorageResult<Vec<StoredProver>>;

    /// Save a proof result
    async fn save_result(&self, result: StoredResult) -> StorageResult<()>;

    /// Get all results for a task
    async fn get_results_for_task(&self, task_id: Uuid) -> StorageResult<Vec<StoredResult>>;

    /// Update task status
    async fn update_task_status(&self, task_id: Uuid, status: TaskStatus) -> StorageResult<()>;

    /// Update prover heartbeat
    async fn update_prover_heartbeat(&self, prover_id: Uuid) -> StorageResult<()>;
}

/// In-memory storage implementation for development and testing
#[derive(Debug)]
pub struct InMemoryStorage {
    tasks: Arc<Mutex<HashMap<Uuid, StoredTask>>>,
    provers: Arc<Mutex<HashMap<Uuid, StoredProver>>>,
    results: Arc<Mutex<HashMap<Uuid, StoredResult>>>,
}

impl Default for InMemoryStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryStorage {
    /// Create a new in-memory storage instance
    pub fn new() -> Self {
        Self {
            tasks: Arc::new(Mutex::new(HashMap::new())),
            provers: Arc::new(Mutex::new(HashMap::new())),
            results: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Get total number of tasks
    pub async fn task_count(&self) -> usize {
        self.tasks.lock().await.len()
    }

    /// Get total number of provers
    pub async fn prover_count(&self) -> usize {
        self.provers.lock().await.len()
    }

    /// Get total number of results
    pub async fn result_count(&self) -> usize {
        self.results.lock().await.len()
    }

    /// Clear all storage (useful for testing)
    pub async fn clear(&self) {
        self.tasks.lock().await.clear();
        self.provers.lock().await.clear();
        self.results.lock().await.clear();
    }
}

#[async_trait]
impl StorageBackend for InMemoryStorage {
    async fn save_task(&self, task: StoredTask) -> StorageResult<()> {
        let mut tasks = self.tasks.lock().await;
        tasks.insert(task.task_id, task);
        Ok(())
    }

    async fn get_task(&self, task_id: Uuid) -> StorageResult<StoredTask> {
        let tasks = self.tasks.lock().await;
        tasks
            .get(&task_id)
            .cloned()
            .ok_or(StorageError::TaskNotFound(task_id))
    }

    async fn list_pending_tasks(&self) -> StorageResult<Vec<StoredTask>> {
        let tasks = self.tasks.lock().await;
        let mut pending: Vec<_> = tasks
            .values()
            .filter(|t| matches!(t.status, TaskStatus::Pending))
            .cloned()
            .collect();
        // Sort by priority descending
        pending.sort_by(|a, b| b.priority.cmp(&a.priority));
        Ok(pending)
    }

    async fn save_prover(&self, prover: StoredProver) -> StorageResult<()> {
        let mut provers = self.provers.lock().await;
        provers.insert(prover.prover_id, prover);
        Ok(())
    }

    async fn get_prover(&self, prover_id: Uuid) -> StorageResult<StoredProver> {
        let provers = self.provers.lock().await;
        provers
            .get(&prover_id)
            .cloned()
            .ok_or(StorageError::ProverNotFound(prover_id))
    }

    async fn list_active_provers(&self) -> StorageResult<Vec<StoredProver>> {
        let provers = self.provers.lock().await;
        let active: Vec<_> = provers
            .values()
            .filter(|p| p.is_active)
            .cloned()
            .collect();
        Ok(active)
    }

    async fn save_result(&self, result: StoredResult) -> StorageResult<()> {
        let mut results = self.results.lock().await;
        results.insert(result.result_id, result);
        Ok(())
    }

    async fn get_results_for_task(&self, task_id: Uuid) -> StorageResult<Vec<StoredResult>> {
        let results = self.results.lock().await;
        let task_results: Vec<_> = results
            .values()
            .filter(|r| r.task_id == task_id)
            .cloned()
            .collect();
        Ok(task_results)
    }

    async fn update_task_status(&self, task_id: Uuid, status: TaskStatus) -> StorageResult<()> {
        let mut tasks = self.tasks.lock().await;
        if let Some(task) = tasks.get_mut(&task_id) {
            task.status = status;
            task.updated_at = Utc::now();
            Ok(())
        } else {
            Err(StorageError::TaskNotFound(task_id))
        }
    }

    async fn update_prover_heartbeat(&self, prover_id: Uuid) -> StorageResult<()> {
        let mut provers = self.provers.lock().await;
        if let Some(prover) = provers.get_mut(&prover_id) {
            prover.last_seen = Utc::now();
            Ok(())
        } else {
            Err(StorageError::ProverNotFound(prover_id))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_task(priority: u32) -> StoredTask {
        StoredTask {
            task_id: Uuid::new_v4(),
            privacy_level: 3,
            circuit_data: vec![1, 2, 3],
            reward_amount: 100,
            requester: "test_requester".to_string(),
            status: TaskStatus::Pending,
            priority,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    fn create_test_prover(is_active: bool) -> StoredProver {
        StoredProver {
            prover_id: Uuid::new_v4(),
            supported_levels: vec![1, 2, 3],
            is_active,
            last_seen: Utc::now(),
            registered_at: Utc::now(),
            capacity: 5,
        }
    }

    fn create_test_result(task_id: Uuid, prover_id: Uuid) -> StoredResult {
        StoredResult {
            result_id: Uuid::new_v4(),
            task_id,
            prover_id,
            proof: vec![0xDE, 0xAD, 0xBE, 0xEF],
            submitted_at: Utc::now(),
            verified: None,
        }
    }

    #[tokio::test]
    async fn test_task_crud() {
        let storage = InMemoryStorage::new();
        let task = create_test_task(10);
        let task_id = task.task_id;

        // Save
        storage.save_task(task.clone()).await.unwrap();
        assert_eq!(storage.task_count().await, 1);

        // Get
        let retrieved = storage.get_task(task_id).await.unwrap();
        assert_eq!(retrieved.task_id, task_id);
        assert_eq!(retrieved.priority, 10);

        // Update status
        storage
            .update_task_status(task_id, TaskStatus::Assigned { prover_id: Uuid::new_v4() })
            .await
            .unwrap();
        let updated = storage.get_task(task_id).await.unwrap();
        assert!(matches!(updated.status, TaskStatus::Assigned { .. }));
    }

    #[tokio::test]
    async fn test_task_not_found() {
        let storage = InMemoryStorage::new();
        let result = storage.get_task(Uuid::new_v4()).await;
        assert!(matches!(result, Err(StorageError::TaskNotFound(_))));
    }

    #[tokio::test]
    async fn test_list_pending_tasks_sorted_by_priority() {
        let storage = InMemoryStorage::new();

        let low_priority = create_test_task(1);
        let high_priority = create_test_task(100);
        let medium_priority = create_test_task(50);

        storage.save_task(low_priority).await.unwrap();
        storage.save_task(high_priority).await.unwrap();
        storage.save_task(medium_priority).await.unwrap();

        let pending = storage.list_pending_tasks().await.unwrap();
        assert_eq!(pending.len(), 3);
        assert_eq!(pending[0].priority, 100);
        assert_eq!(pending[1].priority, 50);
        assert_eq!(pending[2].priority, 1);
    }

    #[tokio::test]
    async fn test_prover_crud() {
        let storage = InMemoryStorage::new();
        let prover = create_test_prover(true);
        let prover_id = prover.prover_id;

        // Save
        storage.save_prover(prover.clone()).await.unwrap();
        assert_eq!(storage.prover_count().await, 1);

        // Get
        let retrieved = storage.get_prover(prover_id).await.unwrap();
        assert_eq!(retrieved.prover_id, prover_id);

        // Update heartbeat
        let old_last_seen = retrieved.last_seen;
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        storage.update_prover_heartbeat(prover_id).await.unwrap();
        let updated = storage.get_prover(prover_id).await.unwrap();
        assert!(updated.last_seen > old_last_seen);
    }

    #[tokio::test]
    async fn test_list_active_provers() {
        let storage = InMemoryStorage::new();

        let active1 = create_test_prover(true);
        let active2 = create_test_prover(true);
        let inactive = create_test_prover(false);

        storage.save_prover(active1).await.unwrap();
        storage.save_prover(active2).await.unwrap();
        storage.save_prover(inactive).await.unwrap();

        let active_provers = storage.list_active_provers().await.unwrap();
        assert_eq!(active_provers.len(), 2);
    }

    #[tokio::test]
    async fn test_result_crud() {
        let storage = InMemoryStorage::new();
        let task_id = Uuid::new_v4();
        let prover_id = Uuid::new_v4();

        let result1 = create_test_result(task_id, prover_id);
        let result2 = create_test_result(task_id, Uuid::new_v4());
        let other_result = create_test_result(Uuid::new_v4(), prover_id);

        storage.save_result(result1).await.unwrap();
        storage.save_result(result2).await.unwrap();
        storage.save_result(other_result).await.unwrap();

        let task_results = storage.get_results_for_task(task_id).await.unwrap();
        assert_eq!(task_results.len(), 2);
    }

    #[tokio::test]
    async fn test_clear_storage() {
        let storage = InMemoryStorage::new();

        storage.save_task(create_test_task(1)).await.unwrap();
        storage.save_prover(create_test_prover(true)).await.unwrap();
        storage.save_result(create_test_result(Uuid::new_v4(), Uuid::new_v4())).await.unwrap();

        assert_eq!(storage.task_count().await, 1);
        assert_eq!(storage.prover_count().await, 1);
        assert_eq!(storage.result_count().await, 1);

        storage.clear().await;

        assert_eq!(storage.task_count().await, 0);
        assert_eq!(storage.prover_count().await, 0);
        assert_eq!(storage.result_count().await, 0);
    }
}
