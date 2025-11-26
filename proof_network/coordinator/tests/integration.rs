//! Integration tests for the NexusZero Distributed Proof Generation Network.
//!
//! These tests verify the end-to-end flow of:
//! 1. Task submission to coordinator
//! 2. Prover registration
//! 3. Task assignment via scheduler
//! 4. Proof generation
//! 5. Result submission and verification

use chrono::Utc;
use coordinator::{
    quality::{ProofResult, QualityVerifier, VerificationResult},
    scheduler::TaskScheduler,
    storage::{InMemoryStorage, StorageBackend, StoredProver, StoredTask, TaskStatus},
};
use marketplace::{
    auction::{AuctionConfig, AuctionEngine},
    order_book::{Ask, Bid, BidAskBook, Order, OrderBook},
    reputation::ReputationTracker,
};
use prover_node::{cpu::CpuProver, gpu::GpuProver, ProofTask, ProverConfig};
use std::sync::Arc;
use uuid::Uuid;

/// Test helper to create a prover config.
fn create_test_prover_config() -> ProverConfig {
    ProverConfig {
        node_id: Uuid::new_v4(),
        gpu_enabled: false,
        max_concurrent_proofs: 4,
        supported_privacy_levels: vec![1, 2, 3, 4],
        coordinator_url: "http://localhost:8080".to_string(),
        reward_address: "0xTestProver".to_string(),
    }
}

/// Test helper to create a proof task.
fn create_test_task(privacy_level: u8) -> ProofTask {
    ProofTask {
        task_id: Uuid::new_v4(),
        privacy_level,
        circuit_data: vec![1, 2, 3, 4, 5, 6, 7, 8],
        reward_amount: 100,
        deadline: Utc::now() + chrono::Duration::hours(1),
        requester: "test-requester".to_string(),
    }
}

#[tokio::test]
async fn test_cpu_prover_generates_valid_proof() {
    let prover = CpuProver::new();
    let task = create_test_task(2);

    let proof = prover
        .generate_proof(&task.circuit_data, task.privacy_level)
        .await
        .expect("Proof generation should succeed");

    // SHA-256 output is 32 bytes
    assert_eq!(proof.len(), 32);

    // Same input should produce same proof (deterministic)
    let proof2 = prover
        .generate_proof(&task.circuit_data, task.privacy_level)
        .await
        .unwrap();
    assert_eq!(proof, proof2);

    // Different privacy level should produce different proof
    let proof3 = prover
        .generate_proof(&task.circuit_data, task.privacy_level + 1)
        .await
        .unwrap();
    assert_ne!(proof, proof3);
}

#[tokio::test]
async fn test_gpu_prover_generates_valid_proof() {
    let prover = GpuProver::new().expect("GPU prover creation should succeed");
    let task = create_test_task(2);

    // GPU prover should work even without real GPU (uses simulation)
    let proof = prover
        .generate_proof(&task.circuit_data, task.privacy_level)
        .await
        .expect("Proof generation should succeed");

    // GPU prover produces 256-byte proof (8 * 32-byte hashes)
    assert_eq!(proof.len(), 256);
}

#[tokio::test]
async fn test_order_book_matching() {
    let mut book = BidAskBook::new();

    // Add some bids (task requesters)
    book.add_bid(Bid {
        bid_id: Uuid::new_v4(),
        requester_id: "requester1".to_string(),
        max_price: 150,
        task_priority: 5,
        deadline: Utc::now() + chrono::Duration::hours(2),
    });

    book.add_bid(Bid {
        bid_id: Uuid::new_v4(),
        requester_id: "requester2".to_string(),
        max_price: 100,
        task_priority: 3,
        deadline: Utc::now() + chrono::Duration::hours(4),
    });

    // Add some asks (provers)
    book.add_ask(Ask {
        ask_id: Uuid::new_v4(),
        prover_id: "prover1".to_string(),
        min_price: 80,
        capacity: 5,
        reputation_score: 0.95,
    });

    book.add_ask(Ask {
        ask_id: Uuid::new_v4(),
        prover_id: "prover2".to_string(),
        min_price: 120,
        capacity: 10,
        reputation_score: 0.85,
    });

    // Match bids with asks
    let matches = book.match_bids_asks();

    // Should have at least one match (bid at 150 can match ask at 80 or 120)
    assert!(!matches.is_empty());

    // All matches should have agreed price between bid max and ask min
    for m in &matches {
        assert!(m.agreed_price >= 80 && m.agreed_price <= 150);
    }
}

#[tokio::test]
async fn test_auction_legacy() {
    let mut order_book = OrderBook::new();

    // Add provers with different prices
    order_book.add_order(Order {
        id: Uuid::new_v4(),
        prover_id: "cheap-prover".to_string(),
        price: 50,
        capacity: 10,
    });

    order_book.add_order(Order {
        id: Uuid::new_v4(),
        prover_id: "expensive-prover".to_string(),
        price: 100,
        capacity: 10,
    });

    // Run auction (legacy price-only method)
    let winners = AuctionEngine::run_auction(&order_book, 3);

    // Should have winners
    assert!(!winners.is_empty());
}

#[tokio::test]
async fn test_auction_with_reputation() {
    let mut order_book = OrderBook::new();
    let mut reputation = ReputationTracker::new();

    // Add provers
    let order1_id = Uuid::new_v4();
    let order2_id = Uuid::new_v4();

    order_book.add_order(Order {
        id: order1_id,
        prover_id: "cheap-low-rep".to_string(),
        price: 50,
        capacity: 10,
    });

    order_book.add_order(Order {
        id: order2_id,
        prover_id: "expensive-high-rep".to_string(),
        price: 100,
        capacity: 10,
    });

    // Create reputation tracker with different scores
    reputation.update_reputation("cheap-low-rep", true, 0.7, 500);
    reputation.update_reputation("expensive-high-rep", true, 0.95, 200);

    // Run auction with reputation weighting
    let config = AuctionConfig {
        price_weight: 0.7,
        reputation_weight: 0.3,
    };
    let engine = AuctionEngine::with_config(config);
    let result = engine.run_auction_with_reputation(&order_book, &reputation, 3);

    // Should have winners
    assert!(!result.winners.is_empty());
}

#[tokio::test]
async fn test_reputation_tracking() {
    let mut tracker = ReputationTracker::new();
    let prover_id = "test-prover";

    // Initial state - no reputation
    assert!(tracker.get_reputation(prover_id).is_none());

    // Add some successful proofs
    for i in 0..5 {
        tracker.update_reputation(prover_id, true, 0.9 + (i as f64 * 0.01), 100);
    }

    let rep = tracker
        .get_reputation(prover_id)
        .expect("Should have reputation");
    assert_eq!(rep.total_proofs, 5);
    assert_eq!(rep.successful_proofs, 5);
    assert_eq!(rep.success_rate(), 1.0);

    // Add a failure
    tracker.update_reputation(prover_id, false, 0.0, 0);

    let rep = tracker.get_reputation(prover_id).unwrap();
    assert_eq!(rep.total_proofs, 6);
    assert_eq!(rep.successful_proofs, 5);
    assert!((rep.success_rate() - 5.0 / 6.0).abs() < 0.001);
}

#[tokio::test]
async fn test_quality_verification() {
    let verifier = QualityVerifier::default();

    // Valid proof (32+ bytes, not all zeros)
    let valid_proof = vec![
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        26, 27, 28, 29, 30, 31, 32,
    ];

    let proof_result = ProofResult {
        result_id: Uuid::new_v4(),
        task_id: Uuid::new_v4(),
        prover_id: Uuid::new_v4(),
        proof: valid_proof.clone(),
        metadata: None,
    };
    let result = verifier.verify_proof(&proof_result);
    assert!(matches!(result, VerificationResult::Pass { .. }));

    // Invalid proof (too short)
    let short_result = ProofResult {
        result_id: Uuid::new_v4(),
        task_id: Uuid::new_v4(),
        prover_id: Uuid::new_v4(),
        proof: vec![1, 2, 3],
        metadata: None,
    };
    let result = verifier.verify_proof(&short_result);
    assert!(matches!(result, VerificationResult::Fail { .. }));

    // Invalid proof (all zeros)
    let zero_result = ProofResult {
        result_id: Uuid::new_v4(),
        task_id: Uuid::new_v4(),
        prover_id: Uuid::new_v4(),
        proof: vec![0; 32],
        metadata: None,
    };
    let result = verifier.verify_proof(&zero_result);
    assert!(matches!(result, VerificationResult::Fail { .. }));
}

#[tokio::test]
async fn test_task_scheduler_integration() {
    let storage = Arc::new(InMemoryStorage::new());
    let scheduler = TaskScheduler::new(storage.clone());

    // Create a task
    let task = StoredTask {
        task_id: Uuid::new_v4(),
        privacy_level: 2,
        circuit_data: vec![1, 2, 3],
        reward_amount: 100,
        requester: "test".to_string(),
        status: TaskStatus::Pending,
        priority: 5,
        created_at: Utc::now(),
        updated_at: Utc::now(),
    };
    storage.save_task(task.clone()).await.unwrap();

    // Register a prover
    let prover = StoredProver {
        prover_id: Uuid::new_v4(),
        supported_levels: vec![1, 2, 3],
        is_active: true,
        last_seen: Utc::now(),
        registered_at: Utc::now(),
        capacity: 4,
    };
    storage.save_prover(prover.clone()).await.unwrap();

    // Schedule the task
    let result = scheduler.schedule_task(task.task_id).await;
    assert!(result.is_ok());
    let assignment = result.unwrap();
    assert_eq!(assignment.task_id, task.task_id);
    assert_eq!(assignment.prover_id, prover.prover_id);

    // Task should now be assigned
    let updated_task = storage.get_task(task.task_id).await.unwrap();
    assert!(matches!(
        updated_task.status,
        TaskStatus::Assigned { prover_id: _ }
    ));
}

#[tokio::test]
async fn test_full_proof_lifecycle() {
    // This test simulates the complete flow:
    // 1. Create coordinator state
    // 2. Register prover
    // 3. Submit task
    // 4. Assign task to prover
    // 5. Generate proof
    // 6. Verify and submit result

    // Setup coordinator
    let storage = Arc::new(InMemoryStorage::new());

    // Register prover
    let prover_id = Uuid::new_v4();
    let prover = StoredProver {
        prover_id,
        supported_levels: vec![1, 2, 3],
        is_active: true,
        last_seen: Utc::now(),
        registered_at: Utc::now(),
        capacity: 4,
    };
    storage.save_prover(prover).await.unwrap();

    // Submit task
    let task_id = Uuid::new_v4();
    let circuit_data = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let task = StoredTask {
        task_id,
        privacy_level: 2,
        circuit_data: circuit_data.clone(),
        reward_amount: 100,
        requester: "test-requester".to_string(),
        status: TaskStatus::Pending,
        priority: 5,
        created_at: Utc::now(),
        updated_at: Utc::now(),
    };
    storage.save_task(task).await.unwrap();

    // Assign task using scheduler
    let scheduler = TaskScheduler::new(storage.clone());
    let assignment = scheduler.schedule_task(task_id).await.unwrap();
    assert_eq!(assignment.prover_id, prover_id);

    // Generate proof using CPU prover
    let cpu_prover = CpuProver::new();
    let proof = cpu_prover.generate_proof(&circuit_data, 2).await.unwrap();
    assert_eq!(proof.len(), 32);

    // Verify proof
    let verifier = QualityVerifier::default();
    let proof_result = ProofResult {
        result_id: Uuid::new_v4(),
        task_id,
        prover_id,
        proof,
        metadata: None,
    };
    let verification = verifier.verify_proof(&proof_result);
    assert!(matches!(verification, VerificationResult::Pass { .. }));

    // Complete task
    storage
        .update_task_status(
            task_id,
            TaskStatus::Completed {
                prover_id,
                completed_at: Utc::now(),
            },
        )
        .await
        .unwrap();

    // Verify final state
    let final_task = storage.get_task(task_id).await.unwrap();
    assert!(matches!(
        final_task.status,
        TaskStatus::Completed {
            prover_id: _,
            completed_at: _
        }
    ));
}

#[tokio::test]
async fn test_gpu_vs_cpu_provers() {
    let cpu_prover = CpuProver::new();
    let gpu_prover = GpuProver::new().unwrap();

    let circuit_data = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let privacy_level = 2;

    // CPU proof
    let cpu_proof = cpu_prover
        .generate_proof(&circuit_data, privacy_level)
        .await
        .unwrap();

    // GPU proof
    let gpu_proof = gpu_prover
        .generate_proof(&circuit_data, privacy_level)
        .await
        .unwrap();

    // Different sizes (GPU produces larger proof)
    assert_eq!(cpu_proof.len(), 32);
    assert_eq!(gpu_proof.len(), 256);

    // Both should pass verification
    let verifier = QualityVerifier::default();

    let cpu_result = ProofResult {
        result_id: Uuid::new_v4(),
        task_id: Uuid::new_v4(),
        prover_id: Uuid::new_v4(),
        proof: cpu_proof,
        metadata: None,
    };
    assert!(verifier.verify_proof(&cpu_result).is_pass());

    let gpu_result = ProofResult {
        result_id: Uuid::new_v4(),
        task_id: Uuid::new_v4(),
        prover_id: Uuid::new_v4(),
        proof: gpu_proof,
        metadata: None,
    };
    assert!(verifier.verify_proof(&gpu_result).is_pass());
}
