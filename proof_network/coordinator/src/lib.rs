//! # Proof Network Coordinator
//!
//! The central REST API server for the NexusZero Distributed Proof Generation Network (DPGN).
//!
//! ## Overview
//!
//! The coordinator is the central orchestration component that:
//! - Accepts and stores proof task submissions from clients
//! - Manages prover registration and health monitoring
//! - Assigns tasks to available provers based on capabilities
//! - Verifies proof quality and updates reputation scores
//! - Stores completed proof results
//!
//! ## Architecture
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────────┐
//! │                         COORDINATOR                                │
//! ├────────────────────────────────────────────────────────────────────┤
//! │                                                                    │
//! │  ┌──────────────────────────────────────────────────────────────┐ │
//! │  │                      REST API (Axum)                          │ │
//! │  │  POST /tasks            - Submit new proof task               │ │
//! │  │  GET  /tasks/:id/status - Get task status                     │ │
//! │  │  GET  /tasks/assign/:id - Assign task to prover               │ │
//! │  │  POST /tasks/result     - Submit proof result                 │ │
//! │  │  GET  /provers          - List registered provers             │ │
//! │  │  POST /provers/register - Register new prover                 │ │
//! │  │  POST /provers/heartbeat- Prover health check                 │ │
//! │  └──────────────────────────────────────────────────────────────┘ │
//! │                              │                                     │
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐   │
//! │  │  Scheduler  │  │   Quality   │  │   Storage Backend       │   │
//! │  │             │  │  Verifier   │  │  (In-Memory / Persist)  │   │
//! │  └─────────────┘  └─────────────┘  └─────────────────────────┘   │
//! │                                                                    │
//! └────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use proof_network_coordinator::{router, CoordinatorState, SharedState};
//! use std::sync::Arc;
//! use tokio::sync::Mutex;
//!
//! #[tokio::main]
//! async fn main() {
//!     // Initialize state
//!     let state: SharedState = Arc::new(Mutex::new(CoordinatorState::new()));
//!     
//!     // Create router with all endpoints
//!     let app = router(state);
//!     
//!     // Start server
//!     let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await.unwrap();
//!     println!("Coordinator listening on :8080");
//!     axum::serve(listener, app).await.unwrap();
//! }
//! ```
//!
//! ## API Endpoints
//!
//! ### Task Management
//!
//! | Endpoint | Method | Description |
//! |----------|--------|-------------|
//! | `/tasks` | POST | Submit a new proof task |
//! | `/tasks/:task_id/status` | GET | Get task status and assignment info |
//! | `/tasks/assign/:prover_id` | GET | Get next available task for a prover |
//! | `/tasks/result` | POST | Submit completed proof for verification |
//!
//! ### Prover Management
//!
//! | Endpoint | Method | Description |
//! |----------|--------|-------------|
//! | `/provers` | GET | List all active provers |
//! | `/provers/register` | POST | Register a new prover node |
//! | `/provers/heartbeat` | POST | Update prover last-seen timestamp |
//!
//! ## Task Lifecycle
//!
//! ```text
//! Pending ──▶ Assigned ──▶ InProgress ──▶ Completed
//!    │           │              │
//!    │           ▼              ▼
//!    │       (timeout)       Failed
//!    │           │
//!    └───────────┘
//! ```
//!
//! ## Modules
//!
//! - [`quality`] - Proof quality verification and scoring
//! - [`scheduler`] - Task scheduling algorithms
//! - [`storage`] - Storage backends (in-memory and persistent)
//!
//! ## Configuration
//!
//! The coordinator can be configured with custom storage backends:
//!
//! ```rust,ignore
//! use proof_network_coordinator::{CoordinatorState, storage::InMemoryStorage};
//! use std::sync::Arc;
//!
//! let storage = Arc::new(InMemoryStorage::new());
//! let state = CoordinatorState::with_storage(storage);
//! ```
//!
//! ## Error Handling
//!
//! All endpoints return appropriate HTTP status codes:
//! - `200 OK` - Successful operation
//! - `400 Bad Request` - Invalid input
//! - `404 Not Found` - Resource not found
//! - `500 Internal Server Error` - Server error

pub mod quality;
pub mod scheduler;
pub mod storage;

use axum::{
    extract::{Path, State},
    routing::{get, post},
    Json, Router,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::quality::{QualityVerifier, VerificationResult};
use crate::storage::{InMemoryStorage, StorageBackend, StoredProver, StoredTask, TaskStatus};

/// Proof task submission request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofTask {
    pub task_id: Uuid,
    pub privacy_level: u8,
    pub circuit_data: Vec<u8>,
    pub reward_amount: u64,
    pub requester: String,
    /// Optional priority (default: 0)
    #[serde(default)]
    pub priority: u32,
}

/// Prover registration request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterProver {
    pub prover_id: Uuid,
    pub supported_levels: Vec<u8>,
    /// Optional capacity (default: 5)
    #[serde(default = "default_capacity")]
    pub capacity: u32,
}

fn default_capacity() -> u32 {
    5
}

/// Proof result submission
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubmitResult {
    pub task_id: Uuid,
    pub prover_id: Uuid,
    pub proof: Vec<u8>,
}

/// Prover heartbeat request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProverHeartbeat {
    pub prover_id: Uuid,
}

/// Task status response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskStatusResponse {
    pub task_id: Uuid,
    pub status: String,
    pub assigned_prover: Option<Uuid>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Prover info response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProverInfo {
    pub prover_id: Uuid,
    pub supported_levels: Vec<u8>,
    pub is_active: bool,
    pub last_seen: DateTime<Utc>,
    pub capacity: u32,
}

/// Result submission response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubmitResultResponse {
    pub accepted: bool,
    pub verification_result: Option<String>,
    pub quality_score: Option<f64>,
}

/// Coordinator state containing storage and quality verifier
pub struct CoordinatorState {
    /// In-memory storage backend
    pub storage: Arc<InMemoryStorage>,
    /// Quality verifier for proof results
    pub quality_verifier: QualityVerifier,
}

impl Default for CoordinatorState {
    fn default() -> Self {
        Self {
            storage: Arc::new(InMemoryStorage::new()),
            quality_verifier: QualityVerifier::new(),
        }
    }
}

impl CoordinatorState {
    /// Create a new coordinator state
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with custom storage
    pub fn with_storage(storage: Arc<InMemoryStorage>) -> Self {
        Self {
            storage,
            quality_verifier: QualityVerifier::new(),
        }
    }
}

/// Shared state type for the coordinator
pub type SharedState = Arc<Mutex<CoordinatorState>>;

// ============================================================================
// API Handlers
// ============================================================================

/// Submit a new proof task
async fn submit_task(
    State(state): State<SharedState>,
    Json(task): Json<ProofTask>,
) -> Json<Uuid> {
    let s = state.lock().await;
    let stored_task = StoredTask {
        task_id: task.task_id,
        privacy_level: task.privacy_level,
        circuit_data: task.circuit_data,
        reward_amount: task.reward_amount,
        requester: task.requester,
        status: TaskStatus::Pending,
        priority: task.priority,
        created_at: Utc::now(),
        updated_at: Utc::now(),
    };
    let _ = s.storage.save_task(stored_task).await;
    Json(task.task_id)
}

/// Register a new prover
async fn register_prover(
    State(state): State<SharedState>,
    Json(reg): Json<RegisterProver>,
) -> Json<Uuid> {
    let s = state.lock().await;
    let stored_prover = StoredProver {
        prover_id: reg.prover_id,
        supported_levels: reg.supported_levels,
        is_active: true,
        last_seen: Utc::now(),
        registered_at: Utc::now(),
        capacity: reg.capacity,
    };
    let _ = s.storage.save_prover(stored_prover).await;
    Json(reg.prover_id)
}

/// Assign a task to a prover
async fn assign_task(
    State(state): State<SharedState>,
    Path(prover_id): Path<String>,
) -> Json<Option<ProofTask>> {
    let s = state.lock().await;

    // Parse prover_id
    let prover_uuid = match Uuid::parse_str(&prover_id) {
        Ok(id) => id,
        Err(_) => return Json(None),
    };

    // Get pending tasks
    if let Ok(pending) = s.storage.list_pending_tasks().await {
        // Get prover to check capabilities
        if let Ok(prover) = s.storage.get_prover(prover_uuid).await {
            // Find first matching task
            for task in pending {
                if prover.supported_levels.contains(&task.privacy_level) {
                    // Mark as assigned
                    let _ = s
                        .storage
                        .update_task_status(task.task_id, TaskStatus::Assigned { prover_id: prover_uuid })
                        .await;

                    return Json(Some(ProofTask {
                        task_id: task.task_id,
                        privacy_level: task.privacy_level,
                        circuit_data: task.circuit_data,
                        reward_amount: task.reward_amount,
                        requester: task.requester,
                        priority: task.priority,
                    }));
                }
            }
        }
    }

    Json(None)
}

/// Submit a proof result
async fn submit_result(
    State(state): State<SharedState>,
    Json(result): Json<SubmitResult>,
) -> Json<SubmitResultResponse> {
    let s = state.lock().await;

    // Create proof result for verification
    let proof_result = quality::ProofResult {
        result_id: Uuid::new_v4(),
        task_id: result.task_id,
        prover_id: result.prover_id,
        proof: result.proof.clone(),
        metadata: None,
    };

    // Verify the proof
    let verification = s.quality_verifier.verify_proof(&proof_result);

    let (accepted, verification_result, quality_score) = match &verification {
        VerificationResult::Pass { quality_score } => {
            // Save the result
            let stored_result = storage::StoredResult {
                result_id: proof_result.result_id,
                task_id: result.task_id,
                prover_id: result.prover_id,
                proof: result.proof,
                submitted_at: Utc::now(),
                verified: Some(true),
            };
            let _ = s.storage.save_result(stored_result).await;

            // Mark task as completed
            let _ = s
                .storage
                .update_task_status(
                    result.task_id,
                    TaskStatus::Completed {
                        prover_id: result.prover_id,
                        completed_at: Utc::now(),
                    },
                )
                .await;

            tracing::info!(
                "Accepted result for task {} from prover {} (quality: {:.2})",
                result.task_id,
                result.prover_id,
                quality_score
            );

            (true, Some("Pass".to_string()), Some(*quality_score))
        }
        VerificationResult::Fail { reason } => {
            tracing::warn!(
                "Rejected result for task {} from prover {}: {}",
                result.task_id,
                result.prover_id,
                reason
            );
            (false, Some(format!("Fail: {}", reason)), None)
        }
    };

    Json(SubmitResultResponse {
        accepted,
        verification_result,
        quality_score,
    })
}

/// List all registered provers
async fn list_provers(State(state): State<SharedState>) -> Json<Vec<ProverInfo>> {
    let s = state.lock().await;
    let provers = s.storage.list_active_provers().await.unwrap_or_default();

    let prover_infos: Vec<ProverInfo> = provers
        .into_iter()
        .map(|p| ProverInfo {
            prover_id: p.prover_id,
            supported_levels: p.supported_levels,
            is_active: p.is_active,
            last_seen: p.last_seen,
            capacity: p.capacity,
        })
        .collect();

    Json(prover_infos)
}

/// Get task status
async fn get_task_status(
    State(state): State<SharedState>,
    Path(task_id): Path<String>,
) -> Json<Option<TaskStatusResponse>> {
    let s = state.lock().await;

    let task_uuid = match Uuid::parse_str(&task_id) {
        Ok(id) => id,
        Err(_) => return Json(None),
    };

    if let Ok(task) = s.storage.get_task(task_uuid).await {
        let (status_str, assigned_prover) = match &task.status {
            TaskStatus::Pending => ("pending".to_string(), None),
            TaskStatus::Assigned { prover_id } => ("assigned".to_string(), Some(*prover_id)),
            TaskStatus::InProgress { prover_id, .. } => ("in_progress".to_string(), Some(*prover_id)),
            TaskStatus::Completed { prover_id, .. } => ("completed".to_string(), Some(*prover_id)),
            TaskStatus::Failed { reason } => (format!("failed: {}", reason), None),
        };

        return Json(Some(TaskStatusResponse {
            task_id: task.task_id,
            status: status_str,
            assigned_prover,
            created_at: task.created_at,
            updated_at: task.updated_at,
        }));
    }

    Json(None)
}

/// Handle prover heartbeat
async fn prover_heartbeat(
    State(state): State<SharedState>,
    Json(heartbeat): Json<ProverHeartbeat>,
) -> Json<bool> {
    let s = state.lock().await;
    match s.storage.update_prover_heartbeat(heartbeat.prover_id).await {
        Ok(_) => {
            tracing::debug!("Heartbeat received from prover {}", heartbeat.prover_id);
            Json(true)
        }
        Err(_) => Json(false),
    }
}

// ============================================================================
// Router
// ============================================================================

/// Create the coordinator router with all endpoints
pub fn router(state: SharedState) -> Router {
    Router::new()
        .route("/tasks", post(submit_task))
        .route("/tasks/:task_id/status", get(get_task_status))
        .route("/tasks/assign/:prover_id", get(assign_task))
        .route("/tasks/result", post(submit_result))
        .route("/provers", get(list_provers))
        .route("/provers/register", post(register_prover))
        .route("/provers/heartbeat", post(prover_heartbeat))
        .with_state(state)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use http_body_util::BodyExt;
    use tower::util::ServiceExt;

    fn create_test_state() -> SharedState {
        Arc::new(Mutex::new(CoordinatorState::default()))
    }

    #[tokio::test]
    async fn test_submit_task() {
        let state = create_test_state();
        let app = router(state.clone());

        let task = ProofTask {
            task_id: Uuid::new_v4(),
            privacy_level: 3,
            circuit_data: vec![1, 2, 3],
            reward_amount: 100,
            requester: "test".to_string(),
            priority: 10,
        };

        let req = Request::builder()
            .method("POST")
            .uri("/tasks")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&task).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Verify task was stored
        let s = state.lock().await;
        let stored = s.storage.get_task(task.task_id).await.unwrap();
        assert_eq!(stored.priority, 10);
    }

    #[tokio::test]
    async fn test_register_prover() {
        let state = create_test_state();
        let app = router(state.clone());

        let prover = RegisterProver {
            prover_id: Uuid::new_v4(),
            supported_levels: vec![1, 2, 3],
            capacity: 5,
        };

        let req = Request::builder()
            .method("POST")
            .uri("/provers/register")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&prover).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Verify prover was stored
        let s = state.lock().await;
        let stored = s.storage.get_prover(prover.prover_id).await.unwrap();
        assert_eq!(stored.supported_levels, vec![1, 2, 3]);
    }

    #[tokio::test]
    async fn test_list_provers() {
        let state = create_test_state();

        // Register a prover first
        {
            let s = state.lock().await;
            let prover = StoredProver {
                prover_id: Uuid::new_v4(),
                supported_levels: vec![1, 2],
                is_active: true,
                last_seen: Utc::now(),
                registered_at: Utc::now(),
                capacity: 5,
            };
            s.storage.save_prover(prover).await.unwrap();
        }

        let app = router(state.clone());

        let req = Request::builder()
            .method("GET")
            .uri("/provers")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let provers: Vec<ProverInfo> = serde_json::from_slice(&body).unwrap();
        assert_eq!(provers.len(), 1);
    }

    #[tokio::test]
    async fn test_get_task_status() {
        let state = create_test_state();
        let task_id = Uuid::new_v4();

        // Add a task
        {
            let s = state.lock().await;
            let task = StoredTask {
                task_id,
                privacy_level: 3,
                circuit_data: vec![1, 2, 3],
                reward_amount: 100,
                requester: "test".to_string(),
                status: TaskStatus::Pending,
                priority: 10,
                created_at: Utc::now(),
                updated_at: Utc::now(),
            };
            s.storage.save_task(task).await.unwrap();
        }

        let app = router(state.clone());

        let req = Request::builder()
            .method("GET")
            .uri(format!("/tasks/{}/status", task_id))
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let status: Option<TaskStatusResponse> = serde_json::from_slice(&body).unwrap();
        assert!(status.is_some());
        assert_eq!(status.unwrap().status, "pending");
    }

    #[tokio::test]
    async fn test_assign_task() {
        let state = create_test_state();
        let prover_id = Uuid::new_v4();
        let task_id = Uuid::new_v4();

        // Add a task and prover
        {
            let s = state.lock().await;
            let task = StoredTask {
                task_id,
                privacy_level: 3,
                circuit_data: vec![1, 2, 3],
                reward_amount: 100,
                requester: "test".to_string(),
                status: TaskStatus::Pending,
                priority: 10,
                created_at: Utc::now(),
                updated_at: Utc::now(),
            };
            let prover = StoredProver {
                prover_id,
                supported_levels: vec![1, 2, 3, 4, 5],
                is_active: true,
                last_seen: Utc::now(),
                registered_at: Utc::now(),
                capacity: 5,
            };
            s.storage.save_task(task).await.unwrap();
            s.storage.save_prover(prover).await.unwrap();
        }

        let app = router(state.clone());

        let req = Request::builder()
            .method("GET")
            .uri(format!("/tasks/assign/{}", prover_id))
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let assigned_task: Option<ProofTask> = serde_json::from_slice(&body).unwrap();
        assert!(assigned_task.is_some());
        assert_eq!(assigned_task.unwrap().task_id, task_id);
    }

    #[tokio::test]
    async fn test_prover_heartbeat() {
        let state = create_test_state();
        let prover_id = Uuid::new_v4();

        // Register prover first
        {
            let s = state.lock().await;
            let prover = StoredProver {
                prover_id,
                supported_levels: vec![1, 2, 3],
                is_active: true,
                last_seen: Utc::now(),
                registered_at: Utc::now(),
                capacity: 5,
            };
            s.storage.save_prover(prover).await.unwrap();
        }

        let app = router(state.clone());

        let heartbeat = ProverHeartbeat { prover_id };

        let req = Request::builder()
            .method("POST")
            .uri("/provers/heartbeat")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&heartbeat).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let result: bool = serde_json::from_slice(&body).unwrap();
        assert!(result);
    }

    #[tokio::test]
    async fn test_submit_result_with_verification() {
        let state = create_test_state();
        let prover_id = Uuid::new_v4();
        let task_id = Uuid::new_v4();

        // Add a task
        {
            let s = state.lock().await;
            let task = StoredTask {
                task_id,
                privacy_level: 3,
                circuit_data: vec![1, 2, 3],
                reward_amount: 100,
                requester: "test".to_string(),
                status: TaskStatus::Assigned { prover_id },
                priority: 10,
                created_at: Utc::now(),
                updated_at: Utc::now(),
            };
            s.storage.save_task(task).await.unwrap();
        }

        let app = router(state.clone());

        // Submit a valid proof (random bytes, at least 32 bytes)
        let proof: Vec<u8> = (0..256).map(|i| (i * 17) as u8).collect();
        let result = SubmitResult {
            task_id,
            prover_id,
            proof,
        };

        let req = Request::builder()
            .method("POST")
            .uri("/tasks/result")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&result).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let response: SubmitResultResponse = serde_json::from_slice(&body).unwrap();
        assert!(response.accepted);
        assert!(response.quality_score.is_some());
    }

    #[tokio::test]
    async fn test_submit_result_fails_verification() {
        let state = create_test_state();
        let app = router(state.clone());

        // Submit an invalid proof (too short)
        let result = SubmitResult {
            task_id: Uuid::new_v4(),
            prover_id: Uuid::new_v4(),
            proof: vec![1, 2, 3], // Too short
        };

        let req = Request::builder()
            .method("POST")
            .uri("/tasks/result")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&result).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let response: SubmitResultResponse = serde_json::from_slice(&body).unwrap();
        assert!(!response.accepted);
        assert!(response.verification_result.unwrap().contains("Fail"));
    }

    #[tokio::test]
    async fn test_coordinator_state_with_storage() {
        let storage = Arc::new(InMemoryStorage::new());
        let state = CoordinatorState::with_storage(storage.clone());

        // Verify storage is shared
        assert_eq!(storage.task_count().await, 0);

        let task = StoredTask {
            task_id: Uuid::new_v4(),
            privacy_level: 3,
            circuit_data: vec![],
            reward_amount: 100,
            requester: "test".to_string(),
            status: TaskStatus::Pending,
            priority: 0,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        state.storage.save_task(task).await.unwrap();

        assert_eq!(storage.task_count().await, 1);
    }
}
