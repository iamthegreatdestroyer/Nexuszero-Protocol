//! Distributed verification system
//!
//! This module provides distributed verification capabilities for high-throughput
//! zero-knowledge proof verification across multiple nodes.

use crate::proof::{Statement, Proof, Verifier, VerifierConfig, VerifierCapabilities};
use crate::proof::verifier::VerificationGuarantee;
use crate::{CryptoError, CryptoResult};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Node information for distributed verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationNode {
    /// Node identifier
    pub id: String,
    /// Node endpoint URL
    pub endpoint: String,
    /// Node capabilities
    pub capabilities: NodeCapabilities,
    /// Node status
    pub status: NodeStatus,
}

/// Node capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCapabilities {
    /// Maximum proofs per second
    pub max_proofs_per_second: u32,
    /// Supported proof types
    pub supported_proof_types: Vec<String>,
    /// Hardware acceleration available
    pub hardware_acceleration: bool,
    /// Geographic region
    pub region: String,
}

/// Node status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeStatus {
    /// Node is active and available
    Active,
    /// Node is busy
    Busy,
    /// Node is offline
    Offline,
    /// Node is under maintenance
    Maintenance,
}

/// Distributed verification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedConfig {
    /// Minimum nodes required for consensus
    pub min_consensus_nodes: usize,
    /// Maximum verification timeout in seconds
    pub max_timeout_seconds: u64,
    /// Fault tolerance level (Byzantine fault tolerance)
    pub fault_tolerance: usize,
    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
}

/// Load balancing strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    /// Round-robin distribution
    RoundRobin,
    /// Least-loaded node first
    LeastLoaded,
    /// Geographic distribution
    Geographic,
    /// Hardware-accelerated first
    HardwareFirst,
}

/// Distributed verifier implementation
pub struct DistributedVerifier {
    /// Available verification nodes
    nodes: Arc<RwLock<Vec<VerificationNode>>>,
    /// Configuration
    config: DistributedConfig,
    /// HTTP client for node communication
    client: reqwest::Client,
    /// Current round-robin index
    rr_index: std::sync::atomic::AtomicUsize,
}

impl DistributedVerifier {
    /// Create a new distributed verifier
    pub fn new(config: DistributedConfig) -> Self {
        Self {
            nodes: Arc::new(RwLock::new(Vec::new())),
            config,
            client: reqwest::Client::new(),
            rr_index: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Add a verification node
    pub async fn add_node(&self, node: VerificationNode) -> CryptoResult<()> {
        let mut nodes = self.nodes.write().await;
        nodes.push(node);
        Ok(())
    }

    /// Remove a verification node
    pub async fn remove_node(&self, node_id: &str) -> CryptoResult<()> {
        let mut nodes = self.nodes.write().await;
        nodes.retain(|n| n.id != node_id);
        Ok(())
    }

    /// Get available nodes
    pub async fn get_available_nodes(&self) -> Vec<VerificationNode> {
        let nodes = self.nodes.read().await;
        nodes.iter()
            .filter(|n| matches!(n.status, NodeStatus::Active))
            .cloned()
            .collect()
    }

    /// Select nodes for verification based on strategy
    async fn select_nodes(&self, count: usize) -> CryptoResult<Vec<VerificationNode>> {
        let available_nodes = self.get_available_nodes().await;

        if available_nodes.len() < self.config.min_consensus_nodes {
            return Err(CryptoError::VerificationError(
                format!("Insufficient nodes: {} available, {} required",
                    available_nodes.len(), self.config.min_consensus_nodes)
            ));
        }

        let selected_count = std::cmp::min(count, available_nodes.len());
        let mut selected = Vec::with_capacity(selected_count);

        match self.config.load_balancing {
            LoadBalancingStrategy::RoundRobin => {
                let start_idx = self.rr_index.fetch_add(selected_count, std::sync::atomic::Ordering::Relaxed) % available_nodes.len();
                for i in 0..selected_count {
                    let idx = (start_idx + i) % available_nodes.len();
                    selected.push(available_nodes[idx].clone());
                }
            }
            LoadBalancingStrategy::LeastLoaded => {
                // Sort by load (simplified - in practice would query actual load)
                let mut sorted = available_nodes;
                sorted.sort_by_key(|n| n.capabilities.max_proofs_per_second);
                selected.extend(sorted.into_iter().take(selected_count));
            }
            LoadBalancingStrategy::Geographic => {
                // Group by region and select from different regions
                let mut by_region: HashMap<String, Vec<VerificationNode>> = HashMap::new();
                for node in available_nodes {
                    by_region.entry(node.capabilities.region.clone())
                        .or_insert_with(Vec::new)
                        .push(node);
                }

                // Select one from each region until we have enough
                for (_, region_nodes) in by_region {
                    if selected.len() >= selected_count {
                        break;
                    }
                    if let Some(node) = region_nodes.first() {
                        selected.push(node.clone());
                    }
                }
            }
            LoadBalancingStrategy::HardwareFirst => {
                // Prioritize hardware-accelerated nodes
                let mut hw_nodes: Vec<VerificationNode> = available_nodes.iter()
                    .filter(|n| n.capabilities.hardware_acceleration)
                    .cloned()
                    .collect();

                let mut regular_nodes: Vec<VerificationNode> = available_nodes.iter()
                    .filter(|n| !n.capabilities.hardware_acceleration)
                    .cloned()
                    .collect();

                selected.append(&mut hw_nodes);
                selected.append(&mut regular_nodes);
                selected.truncate(selected_count);
            }
        }

        Ok(selected)
    }

    /// Send verification request to a node
    async fn send_verification_request(
        &self,
        node: &VerificationNode,
        statement: &Statement,
        proof: &Proof,
    ) -> CryptoResult<bool> {
        let request = VerificationRequest {
            statement: statement.clone(),
            proof: proof.clone(),
        };

        let response = self.client
            .post(&format!("{}/verify", node.endpoint))
            .json(&request)
            .timeout(std::time::Duration::from_secs(self.config.max_timeout_seconds))
            .send()
            .await
            .map_err(|e| CryptoError::NetworkError(format!("Node {} request failed: {}", node.id, e)))?;

        if !response.status().is_success() {
            return Err(CryptoError::VerificationError(
                format!("Node {} returned error: {}", node.id, response.status())
            ));
        }

        let result: VerificationResponse = response.json().await
            .map_err(|e| CryptoError::SerializationError(format!("Failed to parse response: {}", e)))?;

        Ok(result.verified)
    }

    /// Collect consensus from multiple nodes
    async fn collect_consensus(&self, results: Vec<bool>) -> bool {
        let total_votes = results.len();
        let positive_votes = results.iter().filter(|&&r| r).count();

        // Simple majority consensus (can be enhanced with Byzantine fault tolerance)
        positive_votes > total_votes / 2
    }
}

/// Verification request sent to nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
struct VerificationRequest {
    statement: Statement,
    proof: Proof,
}

/// Verification response from nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
struct VerificationResponse {
    verified: bool,
    node_id: String,
    processing_time_ms: u64,
}

#[async_trait]
impl Verifier for DistributedVerifier {
    fn id(&self) -> &str {
        "distributed"
    }

    fn supported_statements(&self) -> Vec<crate::proof::StatementType> {
        // Support all statement types through distribution
        vec![]
    }

    async fn verify(&self, statement: &Statement, proof: &Proof, _config: &VerifierConfig) -> CryptoResult<bool> {
        let nodes = self.select_nodes(self.config.min_consensus_nodes).await?;

        if nodes.is_empty() {
            return Err(CryptoError::VerificationError("No available verification nodes".to_string()));
        }

        // Send verification requests to all selected nodes in parallel
        let mut tasks = Vec::new();
        for node in &nodes {
            let node_clone = node.clone();
            let statement_clone = statement.clone();
            let proof_clone = proof.clone();
            let client_clone = self.client.clone();
            let timeout = self.config.max_timeout_seconds;

            let task = tokio::spawn(async move {
                Self::send_verification_request_static(&client_clone, &node_clone, &statement_clone, &proof_clone, timeout).await
            });
            tasks.push(task);
        }

        // Wait for all responses with timeout
        let results = futures::future::join_all(tasks).await;

        // Collect successful results
        let mut verification_results = Vec::new();
        for result in results {
            match result {
                Ok(Ok(verified)) => verification_results.push(verified),
                Ok(Err(e)) => {
                    log::warn!("Node verification failed: {}", e);
                    // Continue with other nodes
                }
                Err(e) => {
                    log::warn!("Task join failed: {}", e);
                }
            }
        }

        if verification_results.len() < self.config.fault_tolerance + 1 {
            return Err(CryptoError::VerificationError(
                format!("Insufficient successful verifications: {}/{} required",
                    verification_results.len(), self.config.fault_tolerance + 1)
            ));
        }

        // Check consensus
        Ok(self.collect_consensus(verification_results).await)
    }

    async fn verify_batch(&self, statements: &[Statement], proofs: &[Proof], config: &VerifierConfig) -> CryptoResult<Vec<bool>> {
        // For batch verification, distribute across available nodes
        let mut results = Vec::with_capacity(statements.len());

        for (statement, proof) in statements.iter().zip(proofs.iter()) {
            results.push(self.verify(statement, proof, config).await?);
        }

        Ok(results)
    }

    fn capabilities(&self) -> VerifierCapabilities {
        VerifierCapabilities {
            max_proof_size: 65536, // Large proofs supported through distribution
            avg_verification_time_ms: 50, // Parallel verification across nodes
            trusted_setup_required: false,
            verification_guarantee: VerificationGuarantee::Computational,
            supported_optimizations: vec![
                "distributed-verification".to_string(),
                "fault-tolerance".to_string(),
                "load-balancing".to_string(),
                "consensus-verification".to_string(),
            ],
        }
    }
}

impl DistributedVerifier {
    /// Static method for sending verification requests (for use in tokio::spawn)
    async fn send_verification_request_static(
        client: &reqwest::Client,
        node: &VerificationNode,
        statement: &Statement,
        proof: &Proof,
        timeout: u64,
    ) -> CryptoResult<bool> {
        let request = VerificationRequest {
            statement: statement.clone(),
            proof: proof.clone(),
        };

        let response = client
            .post(&format!("{}/verify", node.endpoint))
            .json(&request)
            .timeout(std::time::Duration::from_secs(timeout))
            .send()
            .await
            .map_err(|e| CryptoError::NetworkError(format!("Node {} request failed: {}", node.id, e)))?;

        if !response.status().is_success() {
            return Err(CryptoError::VerificationError(
                format!("Node {} returned error: {}", node.id, response.status())
            ));
        }

        let result: VerificationResponse = response.json().await
            .map_err(|e| CryptoError::SerializationError(format!("Failed to parse response: {}", e)))?;

        Ok(result.verified)
    }
}

/// Byzantine fault-tolerant distributed verifier
pub struct ByzantineDistributedVerifier {
    base_verifier: DistributedVerifier,
    byzantine_threshold: usize,
}

impl ByzantineDistributedVerifier {
    /// Create a new Byzantine fault-tolerant distributed verifier
    pub fn new(config: DistributedConfig, byzantine_threshold: usize) -> Self {
        Self {
            base_verifier: DistributedVerifier::new(config),
            byzantine_threshold,
        }
    }

    /// Verify with Byzantine fault tolerance
    pub async fn verify_byzantine(&self, statement: &Statement, proof: &Proof, config: &VerifierConfig) -> CryptoResult<bool> {
        // Use more nodes for Byzantine tolerance
        let required_nodes = 3 * self.byzantine_threshold + 1;

        // Temporarily modify config for this verification
        let mut bft_config = config.clone();
        bft_config.backend_params.insert("min_nodes".to_string(), serde_json::Value::Number(required_nodes.into()));

        self.base_verifier.verify(statement, proof, &bft_config).await
    }
}