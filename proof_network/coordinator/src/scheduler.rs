//! Task Scheduler for Coordinator
//!
//! Handles task assignment to provers based on priority and availability.
//! Integrates with the marketplace auction system for optimal prover selection.

use crate::storage::{InMemoryStorage, StorageBackend, StoredTask, TaskStatus};
use chrono::Utc;
use std::sync::Arc;
use uuid::Uuid;

/// Error types for scheduler operations
#[derive(Debug, thiserror::Error)]
pub enum SchedulerError {
    #[error("No provers available")]
    NoProversAvailable,
    #[error("No pending tasks")]
    NoPendingTasks,
    #[error("Task not found: {0}")]
    TaskNotFound(Uuid),
    #[error("Prover not found: {0}")]
    ProverNotFound(Uuid),
    #[error("Task already assigned: {0}")]
    TaskAlreadyAssigned(Uuid),
    #[error("Storage error: {0}")]
    StorageError(String),
}

impl From<crate::storage::StorageError> for SchedulerError {
    fn from(err: crate::storage::StorageError) -> Self {
        SchedulerError::StorageError(err.to_string())
    }
}

/// Result type for scheduler operations
pub type SchedulerResult<T> = Result<T, SchedulerError>;

/// Assignment result containing task and prover information
#[derive(Debug, Clone)]
pub struct TaskAssignment {
    /// The assigned task ID
    pub task_id: Uuid,
    /// The assigned prover ID
    pub prover_id: Uuid,
    /// Assignment timestamp
    pub assigned_at: chrono::DateTime<Utc>,
}

/// Marketplace integration for auction-based prover selection
pub struct MarketplaceIntegration {
    /// Whether to use auction for prover selection
    pub use_auction: bool,
}

impl Default for MarketplaceIntegration {
    fn default() -> Self {
        Self { use_auction: false }
    }
}

impl MarketplaceIntegration {
    /// Create a new marketplace integration
    pub fn new(use_auction: bool) -> Self {
        Self { use_auction }
    }

    /// Find the best prover for a task via auction
    /// Returns the winning prover ID and their score
    pub async fn find_best_prover(
        &self,
        _task: &StoredTask,
        available_provers: Vec<Uuid>,
    ) -> Option<(Uuid, f64)> {
        if available_provers.is_empty() {
            return None;
        }

        if self.use_auction {
            // In a real implementation, this would run an actual auction
            // For now, just return the first available prover with a default score
            Some((available_provers[0], 1.0))
        } else {
            // Simple selection: first available prover
            Some((available_provers[0], 1.0))
        }
    }
}

/// Task scheduler for coordinating proof generation
pub struct TaskScheduler {
    /// Storage backend
    storage: Arc<InMemoryStorage>,
    /// Marketplace integration for auction-based selection
    marketplace: MarketplaceIntegration,
}

impl TaskScheduler {
    /// Create a new task scheduler
    pub fn new(storage: Arc<InMemoryStorage>) -> Self {
        Self {
            storage,
            marketplace: MarketplaceIntegration::default(),
        }
    }

    /// Create a new task scheduler with marketplace integration
    pub fn with_marketplace(storage: Arc<InMemoryStorage>, marketplace: MarketplaceIntegration) -> Self {
        Self { storage, marketplace }
    }

    /// Schedule a task by finding the best prover via auction
    ///
    /// # Arguments
    /// * `task_id` - The ID of the task to schedule
    ///
    /// # Returns
    /// The assignment result if successful
    pub async fn schedule_task(&self, task_id: Uuid) -> SchedulerResult<TaskAssignment> {
        // Get the task
        let task = self.storage.get_task(task_id).await?;

        // Check if already assigned
        if !matches!(task.status, TaskStatus::Pending) {
            return Err(SchedulerError::TaskAlreadyAssigned(task_id));
        }

        // Get available provers
        let active_provers = self.storage.list_active_provers().await?;
        if active_provers.is_empty() {
            return Err(SchedulerError::NoProversAvailable);
        }

        // Filter provers that support the required privacy level
        let suitable_provers: Vec<_> = active_provers
            .into_iter()
            .filter(|p| p.supported_levels.contains(&task.privacy_level))
            .map(|p| p.prover_id)
            .collect();

        if suitable_provers.is_empty() {
            return Err(SchedulerError::NoProversAvailable);
        }

        // Find best prover via marketplace/auction
        let (prover_id, _score) = self
            .marketplace
            .find_best_prover(&task, suitable_provers)
            .await
            .ok_or(SchedulerError::NoProversAvailable)?;

        // Mark task as assigned
        self.mark_task_assigned(task_id, prover_id).await?;

        Ok(TaskAssignment {
            task_id,
            prover_id,
            assigned_at: Utc::now(),
        })
    }

    /// Get the next highest priority unassigned task for a specific prover
    ///
    /// # Arguments
    /// * `prover_id` - The prover requesting a task
    ///
    /// # Returns
    /// The highest priority pending task suitable for the prover
    pub async fn get_next_task_for_prover(&self, prover_id: Uuid) -> SchedulerResult<StoredTask> {
        // Verify prover exists and is active
        let prover = self.storage.get_prover(prover_id).await?;
        if !prover.is_active {
            return Err(SchedulerError::ProverNotFound(prover_id));
        }

        // Get pending tasks (already sorted by priority)
        let pending_tasks = self.storage.list_pending_tasks().await?;

        // Find first task that matches prover's capabilities
        for task in pending_tasks {
            if prover.supported_levels.contains(&task.privacy_level) {
                return Ok(task);
            }
        }

        Err(SchedulerError::NoPendingTasks)
    }

    /// Mark a task as assigned to a specific prover
    ///
    /// # Arguments
    /// * `task_id` - The task to mark as assigned
    /// * `prover_id` - The prover to assign the task to
    pub async fn mark_task_assigned(&self, task_id: Uuid, prover_id: Uuid) -> SchedulerResult<()> {
        // Verify prover exists
        let _ = self.storage.get_prover(prover_id).await?;

        // Update task status
        self.storage
            .update_task_status(task_id, TaskStatus::Assigned { prover_id })
            .await?;

        tracing::info!(
            task_id = %task_id,
            prover_id = %prover_id,
            "Task assigned to prover"
        );

        Ok(())
    }

    /// Mark a task as in progress
    pub async fn mark_task_in_progress(&self, task_id: Uuid, prover_id: Uuid) -> SchedulerResult<()> {
        self.storage
            .update_task_status(
                task_id,
                TaskStatus::InProgress {
                    prover_id,
                    started_at: Utc::now(),
                },
            )
            .await?;
        Ok(())
    }

    /// Mark a task as completed
    pub async fn mark_task_completed(&self, task_id: Uuid, prover_id: Uuid) -> SchedulerResult<()> {
        self.storage
            .update_task_status(
                task_id,
                TaskStatus::Completed {
                    prover_id,
                    completed_at: Utc::now(),
                },
            )
            .await?;
        Ok(())
    }

    /// Mark a task as failed
    pub async fn mark_task_failed(&self, task_id: Uuid, reason: String) -> SchedulerResult<()> {
        self.storage
            .update_task_status(task_id, TaskStatus::Failed { reason })
            .await?;
        Ok(())
    }

    /// Get statistics about current scheduling state
    pub async fn get_stats(&self) -> SchedulerStats {
        let pending_count = self
            .storage
            .list_pending_tasks()
            .await
            .map(|t| t.len())
            .unwrap_or(0);
        let active_provers = self
            .storage
            .list_active_provers()
            .await
            .map(|p| p.len())
            .unwrap_or(0);

        SchedulerStats {
            pending_tasks: pending_count,
            active_provers,
        }
    }
}

/// Scheduler statistics
#[derive(Debug, Clone)]
pub struct SchedulerStats {
    /// Number of pending tasks
    pub pending_tasks: usize,
    /// Number of active provers
    pub active_provers: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{StoredProver, StoredTask, TaskStatus};
    use chrono::Utc;

    fn create_test_storage() -> Arc<InMemoryStorage> {
        Arc::new(InMemoryStorage::new())
    }

    fn create_test_task(priority: u32, privacy_level: u8) -> StoredTask {
        StoredTask {
            task_id: Uuid::new_v4(),
            privacy_level,
            circuit_data: vec![1, 2, 3],
            reward_amount: 100,
            requester: "test".to_string(),
            status: TaskStatus::Pending,
            priority,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    fn create_test_prover(supported_levels: Vec<u8>) -> StoredProver {
        StoredProver {
            prover_id: Uuid::new_v4(),
            supported_levels,
            is_active: true,
            last_seen: Utc::now(),
            registered_at: Utc::now(),
            capacity: 5,
        }
    }

    #[tokio::test]
    async fn test_schedule_task_success() {
        let storage = create_test_storage();
        let scheduler = TaskScheduler::new(storage.clone());

        // Add a task and prover
        let task = create_test_task(10, 3);
        let task_id = task.task_id;
        let prover = create_test_prover(vec![1, 2, 3, 4, 5]);
        let prover_id = prover.prover_id;

        storage.save_task(task).await.unwrap();
        storage.save_prover(prover).await.unwrap();

        // Schedule the task
        let assignment = scheduler.schedule_task(task_id).await.unwrap();
        assert_eq!(assignment.task_id, task_id);
        assert_eq!(assignment.prover_id, prover_id);

        // Verify task status updated
        let updated_task = storage.get_task(task_id).await.unwrap();
        assert!(matches!(updated_task.status, TaskStatus::Assigned { .. }));
    }

    #[tokio::test]
    async fn test_schedule_task_no_provers() {
        let storage = create_test_storage();
        let scheduler = TaskScheduler::new(storage.clone());

        let task = create_test_task(10, 3);
        let task_id = task.task_id;
        storage.save_task(task).await.unwrap();

        let result = scheduler.schedule_task(task_id).await;
        assert!(matches!(result, Err(SchedulerError::NoProversAvailable)));
    }

    #[tokio::test]
    async fn test_schedule_task_no_suitable_prover() {
        let storage = create_test_storage();
        let scheduler = TaskScheduler::new(storage.clone());

        // Task requires privacy level 5
        let task = create_test_task(10, 5);
        let task_id = task.task_id;
        storage.save_task(task).await.unwrap();

        // Prover only supports levels 1-3
        let prover = create_test_prover(vec![1, 2, 3]);
        storage.save_prover(prover).await.unwrap();

        let result = scheduler.schedule_task(task_id).await;
        assert!(matches!(result, Err(SchedulerError::NoProversAvailable)));
    }

    #[tokio::test]
    async fn test_schedule_already_assigned_task() {
        let storage = create_test_storage();
        let scheduler = TaskScheduler::new(storage.clone());

        let prover = create_test_prover(vec![1, 2, 3]);
        storage.save_prover(prover.clone()).await.unwrap();

        let mut task = create_test_task(10, 3);
        task.status = TaskStatus::Assigned { prover_id: prover.prover_id };
        let task_id = task.task_id;
        storage.save_task(task).await.unwrap();

        let result = scheduler.schedule_task(task_id).await;
        assert!(matches!(result, Err(SchedulerError::TaskAlreadyAssigned(_))));
    }

    #[tokio::test]
    async fn test_get_next_task_for_prover() {
        let storage = create_test_storage();
        let scheduler = TaskScheduler::new(storage.clone());

        // Add tasks with different priorities
        let low_priority_task = create_test_task(1, 3);
        let high_priority_task = create_test_task(100, 3);

        storage.save_task(low_priority_task.clone()).await.unwrap();
        storage.save_task(high_priority_task.clone()).await.unwrap();

        // Add prover
        let prover = create_test_prover(vec![3]);
        let prover_id = prover.prover_id;
        storage.save_prover(prover).await.unwrap();

        // Should get highest priority task
        let next_task = scheduler.get_next_task_for_prover(prover_id).await.unwrap();
        assert_eq!(next_task.priority, 100);
    }

    #[tokio::test]
    async fn test_get_next_task_filters_by_capability() {
        let storage = create_test_storage();
        let scheduler = TaskScheduler::new(storage.clone());

        // High priority task at level 5 (prover doesn't support)
        let task_level_5 = create_test_task(100, 5);
        // Low priority task at level 2 (prover supports)
        let task_level_2 = create_test_task(10, 2);

        storage.save_task(task_level_5).await.unwrap();
        storage.save_task(task_level_2.clone()).await.unwrap();

        // Prover only supports levels 1-3
        let prover = create_test_prover(vec![1, 2, 3]);
        let prover_id = prover.prover_id;
        storage.save_prover(prover).await.unwrap();

        // Should get the level 2 task even though it has lower priority
        let next_task = scheduler.get_next_task_for_prover(prover_id).await.unwrap();
        assert_eq!(next_task.privacy_level, 2);
    }

    #[tokio::test]
    async fn test_mark_task_states() {
        let storage = create_test_storage();
        let scheduler = TaskScheduler::new(storage.clone());

        let task = create_test_task(10, 3);
        let task_id = task.task_id;
        storage.save_task(task).await.unwrap();

        let prover = create_test_prover(vec![3]);
        let prover_id = prover.prover_id;
        storage.save_prover(prover).await.unwrap();

        // Assign
        scheduler.mark_task_assigned(task_id, prover_id).await.unwrap();
        let t = storage.get_task(task_id).await.unwrap();
        assert!(matches!(t.status, TaskStatus::Assigned { .. }));

        // In progress
        scheduler.mark_task_in_progress(task_id, prover_id).await.unwrap();
        let t = storage.get_task(task_id).await.unwrap();
        assert!(matches!(t.status, TaskStatus::InProgress { .. }));

        // Complete
        scheduler.mark_task_completed(task_id, prover_id).await.unwrap();
        let t = storage.get_task(task_id).await.unwrap();
        assert!(matches!(t.status, TaskStatus::Completed { .. }));
    }

    #[tokio::test]
    async fn test_mark_task_failed() {
        let storage = create_test_storage();
        let scheduler = TaskScheduler::new(storage.clone());

        let task = create_test_task(10, 3);
        let task_id = task.task_id;
        storage.save_task(task).await.unwrap();

        scheduler
            .mark_task_failed(task_id, "Proof generation timeout".to_string())
            .await
            .unwrap();

        let t = storage.get_task(task_id).await.unwrap();
        if let TaskStatus::Failed { reason } = t.status {
            assert_eq!(reason, "Proof generation timeout");
        } else {
            panic!("Expected Failed status");
        }
    }

    #[tokio::test]
    async fn test_scheduler_stats() {
        let storage = create_test_storage();
        let scheduler = TaskScheduler::new(storage.clone());

        // Add some tasks and provers
        storage.save_task(create_test_task(1, 3)).await.unwrap();
        storage.save_task(create_test_task(2, 3)).await.unwrap();
        storage.save_prover(create_test_prover(vec![3])).await.unwrap();

        let stats = scheduler.get_stats().await;
        assert_eq!(stats.pending_tasks, 2);
        assert_eq!(stats.active_provers, 1);
    }
}
