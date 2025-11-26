//! # Prover Node - DPGN
//!
//! A high-performance prover node implementation for the NexusZero Distributed
//! Proof Generation Network (DPGN).
//!
//! ## Overview
//!
//! The prover node is responsible for:
//! - Registering with the coordinator and advertising capabilities
//! - Receiving proof generation tasks from the network
//! - Generating zero-knowledge proofs using CPU or GPU acceleration
//! - Submitting completed proofs back to the coordinator
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────┐
//! │                    ProverNode                        │
//! ├─────────────────────────────────────────────────────┤
//! │  ┌─────────────┐    ┌─────────────┐                 │
//! │  │ TaskQueue   │───▶│ CpuProver   │                 │
//! │  │ (mpsc)      │    └─────────────┘                 │
//! │  └─────────────┘    ┌─────────────┐                 │
//! │                     │ GpuProver   │ (optional)      │
//! │  ┌─────────────┐    └─────────────┘                 │
//! │  │ Coordinator │                                    │
//! │  │ Client      │ HTTP ──▶ Coordinator API           │
//! │  └─────────────┘                                    │
//! └─────────────────────────────────────────────────────┘
//! ```
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use prover_node::{ProverConfig, ProverNode};
//! use uuid::Uuid;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = ProverConfig {
//!         node_id: Uuid::new_v4(),
//!         gpu_enabled: false,
//!         max_concurrent_proofs: 4,
//!         supported_privacy_levels: vec![1, 2, 3],
//!         coordinator_url: "http://localhost:8080".to_string(),
//!         reward_address: "0x1234...".to_string(),
//!     };
//!
//!     let node = ProverNode::new(config).await?;
//!     node.start().await?;
//!     Ok(())
//! }
//! ```
//!
//! ## Components
//!
//! - [`ProverNode`] - Main prover node struct managing the proof generation lifecycle
//! - [`ProverConfig`] - Configuration for prover node initialization
//! - [`TaskQueue`] - Async task queue for managing incoming proof tasks
//! - [`ProofTask`] - Represents a proof generation task from the network
//! - [`ProofResult`] - Result of proof generation including timing and quality metrics
//! - [`cpu::CpuProver`] - CPU-based proof generation implementation
//! - [`gpu::GpuProver`] - GPU-accelerated proof generation (feature-gated)
//! - [`client::CoordinatorClient`] - HTTP client for coordinator communication
//!
//! ## Error Handling
//!
//! All operations return [`ProverError`] variants for comprehensive error handling:
//! - `GpuInitFailed` - GPU initialization errors
//! - `ProofGenerationFailed` - Proof generation failures
//! - `CoordinatorConnectionFailed` - Network connectivity issues
//! - `TaskQueueError` - Task queue management errors

use serde::{Deserialize, Serialize};
use uuid::Uuid;
use tokio::sync::{mpsc, Mutex};
use std::sync::Arc;
use tracing::info;

pub mod cpu;
pub mod gpu;
pub mod client;

pub use client::{CoordinatorClient, RegisterRequest, RegisterResponse, SubmitResultResponse};
pub use gpu::{GpuProver, GpuDevice, GpuStatus};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProverConfig {
    pub node_id: Uuid,
    pub gpu_enabled: bool,
    pub max_concurrent_proofs: usize,
    pub supported_privacy_levels: Vec<u8>,
    pub coordinator_url: String,
    pub reward_address: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofTask {
    pub task_id: Uuid,
    pub privacy_level: u8,
    pub circuit_data: Vec<u8>,
    pub reward_amount: u64,
    pub deadline: chrono::DateTime<chrono::Utc>,
    pub requester: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofResult {
    pub task_id: Uuid,
    pub prover_id: Uuid,
    pub proof: Vec<u8>,
    pub generation_time_ms: u64,
    pub quality_score: f64,
}

#[derive(Debug, thiserror::Error)]
pub enum ProverError {
    #[error("GPU initialization failed: {0}")]
    GpuInitFailed(String),
    #[error("Proof generation failed: {0}")]
    ProofGenerationFailed(String),
    #[error("Coordinator connection failed: {0}")]
    CoordinatorConnectionFailed(String),
    #[error("Task queue error: {0}")]
    TaskQueueError(String),
}

/// Simple task queue using a bounded mpsc channel.
pub struct TaskQueue {
    sender: mpsc::Sender<ProofTask>,
    receiver: Arc<Mutex<mpsc::Receiver<ProofTask>>>,
}

impl TaskQueue {
    pub fn new(buffer: usize) -> Self {
        let (sender, receiver) = mpsc::channel(buffer);
        Self { sender, receiver: Arc::new(Mutex::new(receiver)) }
    }

    pub async fn push_task(&self, task: ProofTask) -> Result<(), ProverError> {
        self.sender.send(task).await.map_err(|e| ProverError::TaskQueueError(e.to_string()))
    }

    pub async fn next_task(&self) -> Option<ProofTask> {
        let mut r = self.receiver.lock().await;
        r.recv().await
    }
}

pub struct ProverNode {
    pub config: ProverConfig,
    pub task_queue: TaskQueue,
    pub cpu_prover: cpu::CpuProver,
}

impl ProverNode {
    pub async fn new(config: ProverConfig) -> Result<Self, ProverError> {
        let task_queue = TaskQueue::new(config.max_concurrent_proofs + 10);
        let cpu_prover = cpu::CpuProver::new();

        Ok(Self { config, task_queue, cpu_prover })
    }

    pub async fn start(self) -> Result<(), ProverError> {
        let node_id = self.config.node_id;
        info!("Starting ProverNode {}", node_id);

        loop {
            if let Some(task) = self.task_queue.next_task().await {
                // Generate proof then submit
                let res = self.generate_proof(&task).await;
                match res {
                    Ok(result) => {
                        // Submit to coordinator (stub)
                        let _ = self.submit_result(result).await;
                    }
                    Err(e) => { tracing::error!("Proof generation error: {e}"); }
                }
            } else {
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            }
        }
    }

    pub async fn generate_proof(&self, task: &ProofTask) -> Result<ProofResult, ProverError> {
        let start = std::time::Instant::now();

        // Use CPU prover for now
        let proof = self.cpu_prover.generate_proof(&task.circuit_data, task.privacy_level)
            .await
            .map_err(|e| ProverError::ProofGenerationFailed(e.to_string()))?;
        let generation_time_ms = start.elapsed().as_millis() as u64;

        let quality_score = Self::calculate_quality_score(&proof);

        Ok(ProofResult {
            task_id: task.task_id,
            prover_id: self.config.node_id,
            proof,
            generation_time_ms,
            quality_score,
        })
    }

    pub async fn submit_result(&self, result: ProofResult) -> Result<(), ProverError> {
        // HTTP submission to coordinator - omitted for now
        info!("Submitting result for task {} by prover {}", result.task_id, result.prover_id);
        Ok(())
    }

    fn calculate_quality_score(proof: &[u8]) -> f64 {
        let expected_size = 1024usize;
        let actual_size = proof.len();
        if actual_size <= expected_size { 1.0 } else { expected_size as f64 / actual_size as f64 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[tokio::test]
    async fn cpu_prover_generates_proof() {
        let config = ProverConfig {
            node_id: Uuid::new_v4(),
            gpu_enabled: false,
            max_concurrent_proofs: 4,
            supported_privacy_levels: vec![3u8],
            coordinator_url: String::new(),
            reward_address: "rewards".to_string(),
        };

        let prover = ProverNode::new(config).await.unwrap();
        let task = ProofTask {
            task_id: Uuid::new_v4(),
            privacy_level: 3,
            circuit_data: vec![1,2,3,4,5],
            reward_amount: 1000,
            deadline: chrono::Utc::now() + chrono::Duration::seconds(60),
            requester: "test".to_string(),
        };

        let res = prover.generate_proof(&task).await.unwrap();
        assert_eq!(res.task_id, task.task_id);
        assert!(res.quality_score > 0.0);
    }
}
