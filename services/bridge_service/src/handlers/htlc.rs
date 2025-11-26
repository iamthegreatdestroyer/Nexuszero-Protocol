//! HTLC (Hash Time-Locked Contract) handlers

use axum::{
    extract::{Path, State},
    Json,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use validator::Validate;

use crate::db::BridgeRepository;
use crate::error::{BridgeError, BridgeResult};
use crate::models::*;
use crate::state::AppState;
use crate::handlers::metrics;

/// Create a new HTLC
pub async fn create_htlc(
    State(state): State<AppState>,
    Json(request): Json<CreateHtlcRequest>,
) -> BridgeResult<Json<ApiResponse<CreateHtlcResponse>>> {
    request.validate().map_err(BridgeError::from)?;
    
    // Verify transfer exists
    let repo = BridgeRepository::new(state.db.clone());
    let transfer = repo
        .get_transfer(request.transfer_id)
        .await?
        .ok_or(BridgeError::TransferNotFound(request.transfer_id.to_string()))?;
    
    // Verify transfer is in correct state
    if transfer.status != TransferStatus::SourceConfirmed {
        return Err(BridgeError::InvalidTransferState {
            transfer_id: request.transfer_id.to_string(),
            expected: "source_confirmed".to_string(),
            actual: format!("{:?}", transfer.status),
        });
    }
    
    // Get chain configuration
    let chain_config = state.config.get_chain(&request.chain)
        .ok_or(BridgeError::UnsupportedChain(request.chain.clone()))?;
    
    let htlc_contract = chain_config.htlc_contract.clone()
        .unwrap_or_else(|| "0x0000000000000000000000000000000000000000".to_string());
    
    // Generate secret and hash
    let (secret, secret_hash) = state.htlc_manager.generate_secret();
    
    // Calculate timelock
    let timelock = state.htlc_manager.calculate_timelock(Some(request.timelock_seconds));
    
    // Create HTLC record
    let htlc = repo
        .create_htlc(
            request.transfer_id,
            &request.chain,
            &htlc_contract,
            &transfer.source_address,
            &request.receiver_address,
            &request.asset,
            request.amount,
            &secret_hash,
            timelock,
        )
        .await?;
    
    // Update transfer status
    repo.update_transfer_status(
        transfer.id,
        TransferStatus::HtlcCreated,
        None,
    ).await?;
    
    // Record metrics
    metrics::record_htlc_operation(&request.chain, "created");
    metrics::HTLC_ACTIVE
        .with_label_values(&[&request.chain, "locked"])
        .inc();
    
    // TODO: In production, actually deploy HTLC to chain
    
    let response = CreateHtlcResponse {
        htlc_id: htlc.id,
        secret_hash: htlc.secret_hash,
        timelock: htlc.timelock,
        contract_address: htlc.contract_address,
        status: htlc.status,
    };
    
    Ok(Json(ApiResponse::success(response)))
}

/// Get HTLC by ID
pub async fn get_htlc(
    State(state): State<AppState>,
    Path(htlc_id): Path<Uuid>,
) -> BridgeResult<Json<ApiResponse<Htlc>>> {
    let repo = BridgeRepository::new(state.db.clone());
    
    let htlc = repo
        .get_htlc(htlc_id)
        .await?
        .ok_or(BridgeError::HtlcNotFound(htlc_id.to_string()))?;
    
    Ok(Json(ApiResponse::success(htlc)))
}

/// Get HTLCs for a transfer
pub async fn get_htlcs_for_transfer(
    State(state): State<AppState>,
    Path(transfer_id): Path<Uuid>,
) -> BridgeResult<Json<ApiResponse<Vec<Htlc>>>> {
    let repo = BridgeRepository::new(state.db.clone());
    
    // Get both source and destination HTLCs
    let mut htlcs = Vec::new();
    
    // Get transfer to know chains
    let transfer = repo
        .get_transfer(transfer_id)
        .await?
        .ok_or(BridgeError::TransferNotFound(transfer_id.to_string()))?;
    
    if let Some(htlc) = repo.get_htlc_by_transfer(transfer_id, &transfer.source_chain).await? {
        htlcs.push(htlc);
    }
    
    if let Some(htlc) = repo.get_htlc_by_transfer(transfer_id, &transfer.destination_chain).await? {
        htlcs.push(htlc);
    }
    
    Ok(Json(ApiResponse::success(htlcs)))
}

/// Claim an HTLC with secret
#[derive(Debug, Deserialize, Validate)]
pub struct ClaimHtlcRequest {
    /// The secret to claim the HTLC
    #[validate(length(equal = 64))]
    pub secret: String,
}

pub async fn claim_htlc(
    State(state): State<AppState>,
    Path(htlc_id): Path<Uuid>,
    Json(request): Json<ClaimHtlcRequest>,
) -> BridgeResult<Json<ApiResponse<Htlc>>> {
    request.validate().map_err(BridgeError::from)?;
    
    let repo = BridgeRepository::new(state.db.clone());
    
    let htlc = repo
        .get_htlc(htlc_id)
        .await?
        .ok_or(BridgeError::HtlcNotFound(htlc_id.to_string()))?;
    
    // Verify HTLC is in locked state
    if htlc.status != HtlcStatus::Locked {
        if htlc.status == HtlcStatus::Claimed {
            return Err(BridgeError::HtlcAlreadyClaimed(htlc_id.to_string()));
        }
        return Err(BridgeError::InvalidTransferState {
            transfer_id: htlc_id.to_string(),
            expected: "locked".to_string(),
            actual: format!("{:?}", htlc.status),
        });
    }
    
    // Verify secret hash matches
    use sha2::{Sha256, Digest};
    let secret_bytes = hex::decode(&request.secret)
        .map_err(|_| BridgeError::InvalidSecret)?;
    let mut hasher = Sha256::new();
    hasher.update(&secret_bytes);
    let computed_hash = hex::encode(hasher.finalize());
    
    if computed_hash != htlc.secret_hash {
        return Err(BridgeError::InvalidSecret);
    }
    
    // TODO: In production, submit claim transaction to chain
    let claim_tx_hash = format!("0x{:064x}", rand::random::<u64>());
    
    // Update HTLC as claimed
    let updated = repo
        .update_htlc_claimed(htlc_id, &request.secret, &claim_tx_hash)
        .await?;
    
    // Record metrics
    metrics::record_htlc_operation(&htlc.chain, "claimed");
    metrics::HTLC_ACTIVE
        .with_label_values(&[&htlc.chain, "locked"])
        .dec();
    
    Ok(Json(ApiResponse::success(updated)))
}

/// Refund an expired HTLC
pub async fn refund_htlc(
    State(state): State<AppState>,
    Path(htlc_id): Path<Uuid>,
) -> BridgeResult<Json<ApiResponse<Htlc>>> {
    let repo = BridgeRepository::new(state.db.clone());
    
    let htlc = repo
        .get_htlc(htlc_id)
        .await?
        .ok_or(BridgeError::HtlcNotFound(htlc_id.to_string()))?;
    
    // Verify HTLC is in locked state
    if htlc.status != HtlcStatus::Locked && htlc.status != HtlcStatus::Expired {
        if htlc.status == HtlcStatus::Refunded {
            return Err(BridgeError::HtlcAlreadyRefunded(htlc_id.to_string()));
        }
        if htlc.status == HtlcStatus::Claimed {
            return Err(BridgeError::HtlcAlreadyClaimed(htlc_id.to_string()));
        }
        return Err(BridgeError::InvalidTransferState {
            transfer_id: htlc_id.to_string(),
            expected: "locked or expired".to_string(),
            actual: format!("{:?}", htlc.status),
        });
    }
    
    // Verify timelock has expired
    let now = chrono::Utc::now();
    if htlc.timelock > now {
        return Err(BridgeError::HtlcNotExpired(htlc_id.to_string()));
    }
    
    // TODO: In production, submit refund transaction to chain
    let refund_tx_hash = format!("0x{:064x}", rand::random::<u64>());
    
    // Update HTLC as refunded
    let updated = repo
        .update_htlc_refunded(htlc_id, &refund_tx_hash)
        .await?;
    
    // Update associated transfer
    repo.update_transfer_status(
        htlc.transfer_id,
        TransferStatus::Refunded,
        Some("HTLC expired and refunded"),
    ).await?;
    
    // Record metrics
    metrics::record_htlc_operation(&htlc.chain, "refunded");
    metrics::HTLC_ACTIVE
        .with_label_values(&[&htlc.chain, "locked"])
        .dec();
    
    Ok(Json(ApiResponse::success(updated)))
}

/// Verify HTLC on chain
#[derive(Debug, Serialize)]
pub struct HtlcVerification {
    pub htlc_id: Uuid,
    pub onchain_exists: bool,
    pub onchain_status: Option<String>,
    pub onchain_amount: Option<String>,
    pub onchain_timelock: Option<i64>,
    pub matches_record: bool,
    pub verified_at: chrono::DateTime<chrono::Utc>,
}

pub async fn verify_htlc(
    State(state): State<AppState>,
    Path(htlc_id): Path<Uuid>,
) -> BridgeResult<Json<ApiResponse<HtlcVerification>>> {
    let repo = BridgeRepository::new(state.db.clone());
    
    let htlc = repo
        .get_htlc(htlc_id)
        .await?
        .ok_or(BridgeError::HtlcNotFound(htlc_id.to_string()))?;
    
    // TODO: In production, query the chain to verify HTLC state
    // This is a placeholder implementation
    
    let verification = HtlcVerification {
        htlc_id,
        onchain_exists: htlc.lock_tx_hash.is_some(),
        onchain_status: Some(format!("{:?}", htlc.status)),
        onchain_amount: Some(htlc.amount.to_string()),
        onchain_timelock: Some(htlc.timelock.timestamp()),
        matches_record: true,
        verified_at: chrono::Utc::now(),
    };
    
    Ok(Json(ApiResponse::success(verification)))
}
