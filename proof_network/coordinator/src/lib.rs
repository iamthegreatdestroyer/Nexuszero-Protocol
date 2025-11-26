use axum::{routing::{get, post}, Json, Router};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofTask {
    pub task_id: Uuid,
    pub privacy_level: u8,
    pub circuit_data: Vec<u8>,
    pub reward_amount: u64,
    pub requester: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterProver {
    pub prover_id: Uuid,
    pub supported_levels: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubmitResult {
    pub task_id: Uuid,
    pub prover_id: Uuid,
    pub proof: Vec<u8>,
}

#[derive(Default)]
pub struct CoordinatorState {
    pub tasks: Vec<ProofTask>,
    pub provers: Vec<RegisterProver>,
}

pub type SharedState = Arc<Mutex<CoordinatorState>>;

#[axum::debug_handler]
async fn submit_task(axum::Extension(state): axum::Extension<SharedState>, axum::extract::Json(task): axum::extract::Json<ProofTask>) -> Json<Uuid> {
    let mut s = state.lock().await;
    s.tasks.push(task.clone());
    Json(task.task_id)
}

#[axum::debug_handler]
async fn register_prover(axum::Extension(state): axum::Extension<SharedState>, axum::extract::Json(reg): axum::extract::Json<RegisterProver>) -> Json<Uuid> {
    let mut s = state.lock().await;
    s.provers.push(reg.clone());
    Json(reg.prover_id)
}

#[axum::debug_handler]
async fn assign_task(axum::Extension(state): axum::Extension<SharedState>, axum::extract::Path(prover_id): axum::extract::Path<String>) -> Json<Option<ProofTask>> {
    // For simplicity, assign first matching task
    let mut s = state.lock().await;
    if let Some(pos) = s.tasks.iter().position(|t| true) {
        let t = s.tasks.remove(pos);
        return Json(Some(t));
    }
    Json(None)
}

#[axum::debug_handler]
async fn submit_result(axum::Extension(state): axum::Extension<SharedState>, axum::extract::Json(result): axum::extract::Json<SubmitResult>) -> Json<bool> {
    // Accept the result – in real world we'd verify proof, reward prover, etc.
    tracing::info!("Received result for task {} from prover {}", result.task_id, result.prover_id);
    Json(true)
}

pub fn router(state: SharedState) -> Router {
    let s1 = state.clone();
    let s2 = state.clone();
    let s3 = state.clone();
    let s4 = state.clone();
    Router::new()
        .route(
            "/tasks",
            post(move |axum::extract::Json(task): axum::extract::Json<ProofTask>| {
                let state = s1.clone();
                async move {
                    let mut s = state.lock().await;
                    s.tasks.push(task.clone());
                    axum::Json(task.task_id)
                }
            }),
        )
        .route(
            "/provers/register",
            post(move |axum::extract::Json(reg): axum::extract::Json<RegisterProver>| {
                let state = s2.clone();
                async move {
                    let mut s = state.lock().await;
                    s.provers.push(reg.clone());
                    axum::Json(reg.prover_id)
                }
            }),
        )
        .route(
            "/tasks/assign/:prover_id",
            get(move |axum::extract::Path(prover_id): axum::extract::Path<String>| {
                let state = s3.clone();
                async move {
                    let mut s = state.lock().await;
                    if let Some(pos) = s.tasks.iter().position(|t| true) {
                        let t = s.tasks.remove(pos);
                        axum::Json(Some(t))
                    } else {
                        axum::Json(None)
                    }
                }
            }),
        )
        .route(
            "/tasks/result",
            post(move |axum::extract::Json(result): axum::extract::Json<SubmitResult>| {
                let state = s4.clone();
                async move {
                    tracing::info!("Received result for task {} from prover {}", result.task_id, result.prover_id);
                    axum::Json(true)
                }
            }),
        )
        .layer(axum::Extension(state))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tower::util::ServiceExt; // for `oneshot`
    use uuid::Uuid;

    #[tokio::test]
    async fn coordinator_routes_work() {
        let state = Arc::new(Mutex::new(CoordinatorState::default()));
        let app = router(state.clone());

        let task = ProofTask { task_id: Uuid::new_v4(), privacy_level: 3, circuit_data: vec![1,2], reward_amount: 10, requester: "test".to_string() };

        let req = Request::builder()
            .method("POST")
            .uri("/tasks")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&task).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), 200);
    }
}
