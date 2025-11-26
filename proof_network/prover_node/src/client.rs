//! HTTP client for coordinator communication.
//!
//! Provides async HTTP client functionality for prover nodes to:
//! - Register with the coordinator
//! - Request task assignments
//! - Submit proof results
//! - Send heartbeat signals

use super::{ProofResult, ProofTask, ProverConfig, ProverError};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

/// Registration request sent to coordinator.
#[derive(Debug, Clone, Serialize)]
pub struct RegisterRequest {
    pub prover_id: String,
    pub supported_privacy_levels: Vec<u8>,
    pub max_concurrent_proofs: usize,
    pub reward_address: String,
    pub gpu_enabled: bool,
}

/// Registration response from coordinator.
#[derive(Debug, Clone, Deserialize)]
pub struct RegisterResponse {
    pub success: bool,
    pub message: String,
    pub assigned_region: Option<String>,
}

/// Task assignment response from coordinator.
#[derive(Debug, Clone, Deserialize)]
pub struct TaskAssignment {
    pub task: Option<ProofTask>,
    pub retry_after_ms: Option<u64>,
}

/// Result submission response from coordinator.
#[derive(Debug, Clone, Deserialize)]
pub struct SubmitResultResponse {
    pub accepted: bool,
    pub reward_issued: Option<u64>,
    pub quality_bonus: Option<f64>,
    pub message: String,
}

/// Heartbeat response from coordinator.
#[derive(Debug, Clone, Deserialize)]
pub struct HeartbeatResponse {
    pub acknowledged: bool,
    pub pending_tasks: usize,
}

/// HTTP client for coordinator API communication.
pub struct CoordinatorClient {
    /// Base URL of the coordinator
    base_url: String,
    /// Prover node ID
    prover_id: String,
    /// HTTP client instance
    client: reqwest::Client,
    /// Connection timeout in seconds
    timeout_secs: u64,
}

impl CoordinatorClient {
    /// Create a new coordinator client.
    ///
    /// # Arguments
    ///
    /// * `config` - Prover configuration containing coordinator URL and node ID
    pub fn new(config: &ProverConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .unwrap_or_default();

        Self {
            base_url: config.coordinator_url.clone(),
            prover_id: config.node_id.to_string(),
            client,
            timeout_secs: 30,
        }
    }

    /// Create a client with custom timeout.
    pub fn with_timeout(mut self, timeout_secs: u64) -> Self {
        self.timeout_secs = timeout_secs;
        self
    }

    /// Register this prover with the coordinator.
    ///
    /// Must be called before requesting tasks.
    pub async fn register(&self, config: &ProverConfig) -> Result<RegisterResponse, ProverError> {
        let url = format!("{}/provers/register", self.base_url);
        
        let request = RegisterRequest {
            prover_id: config.node_id.to_string(),
            supported_privacy_levels: config.supported_privacy_levels.clone(),
            max_concurrent_proofs: config.max_concurrent_proofs,
            reward_address: config.reward_address.clone(),
            gpu_enabled: config.gpu_enabled,
        };

        debug!("Registering prover {} with coordinator", config.node_id);

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| ProverError::CoordinatorConnectionFailed(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(ProverError::CoordinatorConnectionFailed(format!(
                "Registration failed: {} - {}",
                status, body
            )));
        }

        let result: RegisterResponse = response
            .json()
            .await
            .map_err(|e| ProverError::CoordinatorConnectionFailed(e.to_string()))?;

        if result.success {
            info!(
                "Prover {} registered successfully, region: {:?}",
                config.node_id, result.assigned_region
            );
        } else {
            warn!("Registration returned failure: {}", result.message);
        }

        Ok(result)
    }

    /// Request a task assignment from the coordinator.
    ///
    /// Returns `None` if no tasks are available.
    pub async fn request_task(&self) -> Result<Option<ProofTask>, ProverError> {
        let url = format!("{}/tasks/assign/{}", self.base_url, self.prover_id);

        debug!("Requesting task for prover {}", self.prover_id);

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| ProverError::CoordinatorConnectionFailed(e.to_string()))?;

        if response.status() == reqwest::StatusCode::NO_CONTENT {
            return Ok(None);
        }

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(ProverError::CoordinatorConnectionFailed(format!(
                "Task request failed: {} - {}",
                status, body
            )));
        }

        let assignment: TaskAssignment = response
            .json()
            .await
            .map_err(|e| ProverError::CoordinatorConnectionFailed(e.to_string()))?;

        if let Some(ref task) = assignment.task {
            info!("Received task {} for prover {}", task.task_id, self.prover_id);
        }

        Ok(assignment.task)
    }

    /// Submit a completed proof result to the coordinator.
    pub async fn submit_result(
        &self,
        result: &ProofResult,
    ) -> Result<SubmitResultResponse, ProverError> {
        let url = format!("{}/tasks/result", self.base_url);

        debug!("Submitting result for task {}", result.task_id);

        let response = self
            .client
            .post(&url)
            .json(result)
            .send()
            .await
            .map_err(|e| ProverError::CoordinatorConnectionFailed(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(ProverError::CoordinatorConnectionFailed(format!(
                "Result submission failed: {} - {}",
                status, body
            )));
        }

        let submit_response: SubmitResultResponse = response
            .json()
            .await
            .map_err(|e| ProverError::CoordinatorConnectionFailed(e.to_string()))?;

        if submit_response.accepted {
            info!(
                "Result for task {} accepted, reward: {:?}",
                result.task_id, submit_response.reward_issued
            );
        } else {
            warn!(
                "Result for task {} rejected: {}",
                result.task_id, submit_response.message
            );
        }

        Ok(submit_response)
    }

    /// Send a heartbeat to the coordinator.
    ///
    /// Should be called periodically to maintain registration.
    pub async fn heartbeat(&self) -> Result<HeartbeatResponse, ProverError> {
        let url = format!("{}/provers/{}/heartbeat", self.base_url, self.prover_id);

        let response = self
            .client
            .post(&url)
            .send()
            .await
            .map_err(|e| ProverError::CoordinatorConnectionFailed(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            return Err(ProverError::CoordinatorConnectionFailed(format!(
                "Heartbeat failed: {}",
                status
            )));
        }

        let heartbeat_response: HeartbeatResponse = response
            .json()
            .await
            .map_err(|e| ProverError::CoordinatorConnectionFailed(e.to_string()))?;

        debug!(
            "Heartbeat acknowledged, {} pending tasks",
            heartbeat_response.pending_tasks
        );

        Ok(heartbeat_response)
    }

    /// Get coordinator base URL.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Get prover ID.
    pub fn prover_id(&self) -> &str {
        &self.prover_id
    }
}

/// Mock coordinator client for testing.
#[cfg(test)]
pub mod mock {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use tokio::sync::Mutex;

    /// Mock client that simulates coordinator responses.
    pub struct MockCoordinatorClient {
        pub registered: Arc<Mutex<bool>>,
        pub tasks_returned: Arc<AtomicUsize>,
        pub results_submitted: Arc<AtomicUsize>,
        pub pending_tasks: Arc<Mutex<Vec<ProofTask>>>,
    }

    impl MockCoordinatorClient {
        pub fn new() -> Self {
            Self {
                registered: Arc::new(Mutex::new(false)),
                tasks_returned: Arc::new(AtomicUsize::new(0)),
                results_submitted: Arc::new(AtomicUsize::new(0)),
                pending_tasks: Arc::new(Mutex::new(Vec::new())),
            }
        }

        pub async fn add_task(&self, task: ProofTask) {
            self.pending_tasks.lock().await.push(task);
        }

        pub async fn register(&self, _config: &ProverConfig) -> Result<RegisterResponse, ProverError> {
            *self.registered.lock().await = true;
            Ok(RegisterResponse {
                success: true,
                message: "Registered".to_string(),
                assigned_region: Some("test-region".to_string()),
            })
        }

        pub async fn request_task(&self) -> Result<Option<ProofTask>, ProverError> {
            let mut tasks = self.pending_tasks.lock().await;
            if let Some(task) = tasks.pop() {
                self.tasks_returned.fetch_add(1, Ordering::SeqCst);
                Ok(Some(task))
            } else {
                Ok(None)
            }
        }

        pub async fn submit_result(&self, _result: &ProofResult) -> Result<SubmitResultResponse, ProverError> {
            self.results_submitted.fetch_add(1, Ordering::SeqCst);
            Ok(SubmitResultResponse {
                accepted: true,
                reward_issued: Some(100),
                quality_bonus: Some(0.05),
                message: "Accepted".to_string(),
            })
        }

        pub async fn heartbeat(&self) -> Result<HeartbeatResponse, ProverError> {
            let pending = self.pending_tasks.lock().await.len();
            Ok(HeartbeatResponse {
                acknowledged: true,
                pending_tasks: pending,
            })
        }
    }

    impl Default for MockCoordinatorClient {
        fn default() -> Self {
            Self::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn test_config() -> ProverConfig {
        ProverConfig {
            node_id: Uuid::new_v4(),
            gpu_enabled: false,
            max_concurrent_proofs: 4,
            supported_privacy_levels: vec![1, 2, 3],
            coordinator_url: "http://localhost:8080".to_string(),
            reward_address: "0x1234".to_string(),
        }
    }

    #[test]
    fn test_client_creation() {
        let config = test_config();
        let client = CoordinatorClient::new(&config);
        
        assert_eq!(client.base_url(), "http://localhost:8080");
        assert_eq!(client.prover_id(), config.node_id.to_string());
    }

    #[test]
    fn test_client_with_timeout() {
        let config = test_config();
        let client = CoordinatorClient::new(&config).with_timeout(60);
        
        assert_eq!(client.timeout_secs, 60);
    }

    #[tokio::test]
    async fn test_mock_client() {
        let mock = mock::MockCoordinatorClient::new();
        let config = test_config();

        // Register
        let reg = mock.register(&config).await.unwrap();
        assert!(reg.success);
        assert!(*mock.registered.lock().await);

        // No tasks initially
        let task = mock.request_task().await.unwrap();
        assert!(task.is_none());

        // Add and request task
        mock.add_task(ProofTask {
            task_id: Uuid::new_v4(),
            privacy_level: 1,
            circuit_data: vec![1, 2, 3],
            reward_amount: 100,
            deadline: chrono::Utc::now() + chrono::Duration::hours(1),
            requester: "test".to_string(),
        }).await;

        let task = mock.request_task().await.unwrap();
        assert!(task.is_some());
        assert_eq!(mock.tasks_returned.load(std::sync::atomic::Ordering::SeqCst), 1);

        // Submit result
        let result = ProofResult {
            task_id: Uuid::new_v4(),
            prover_id: config.node_id,
            proof: vec![0; 32],
            generation_time_ms: 100,
            quality_score: 0.95,
        };
        let submit = mock.submit_result(&result).await.unwrap();
        assert!(submit.accepted);
        assert_eq!(mock.results_submitted.load(std::sync::atomic::Ordering::SeqCst), 1);

        // Heartbeat
        let hb = mock.heartbeat().await.unwrap();
        assert!(hb.acknowledged);
    }
}
