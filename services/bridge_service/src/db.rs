//! Bridge Service Database Repository
//! 
//! Database operations for cross-chain bridge service.

use crate::error::{BridgeError, BridgeResult};
use crate::models::*;
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use sqlx::PgPool;
use uuid::Uuid;

/// Database repository for bridge operations
#[derive(Clone)]
pub struct BridgeRepository {
    pool: PgPool,
}

impl BridgeRepository {
    /// Create new repository
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }
    
    // ========================================================================
    // TRANSFER OPERATIONS
    // ========================================================================
    
    /// Create a new bridge transfer
    pub async fn create_transfer(
        &self,
        user_id: Uuid,
        source_chain: &str,
        source_address: &str,
        destination_chain: &str,
        destination_address: &str,
        asset_symbol: &str,
        asset_address: Option<&str>,
        amount: Decimal,
        bridge_fee: Decimal,
        direction: TransferDirection,
    ) -> BridgeResult<BridgeTransfer> {
        let transfer_id = format!("TXF-{}", Uuid::new_v4().to_string().split('-').next().unwrap().to_uppercase());
        
        let transfer = sqlx::query_as::<_, BridgeTransfer>(
            r#"
            INSERT INTO bridge_transfers (
                transfer_id, user_id,
                source_chain, source_address,
                destination_chain, destination_address,
                asset_symbol, asset_address,
                amount, bridge_fee,
                status, direction,
                initiated_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                'pending', $11, NOW()
            )
            RETURNING *
            "#
        )
        .bind(&transfer_id)
        .bind(user_id)
        .bind(source_chain)
        .bind(source_address)
        .bind(destination_chain)
        .bind(destination_address)
        .bind(asset_symbol)
        .bind(asset_address)
        .bind(amount)
        .bind(bridge_fee)
        .bind(direction)
        .fetch_one(&self.pool)
        .await?;
        
        Ok(transfer)
    }
    
    /// Get transfer by ID
    pub async fn get_transfer(&self, id: Uuid) -> BridgeResult<Option<BridgeTransfer>> {
        let transfer = sqlx::query_as::<_, BridgeTransfer>(
            "SELECT * FROM bridge_transfers WHERE id = $1"
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;
        
        Ok(transfer)
    }
    
    /// Get transfer by transfer_id
    pub async fn get_transfer_by_transfer_id(
        &self,
        transfer_id: &str,
    ) -> BridgeResult<Option<BridgeTransfer>> {
        let transfer = sqlx::query_as::<_, BridgeTransfer>(
            "SELECT * FROM bridge_transfers WHERE transfer_id = $1"
        )
        .bind(transfer_id)
        .fetch_optional(&self.pool)
        .await?;
        
        Ok(transfer)
    }
    
    /// Get transfers for user
    pub async fn get_user_transfers(
        &self,
        user_id: Uuid,
        page: i32,
        page_size: i32,
    ) -> BridgeResult<(Vec<BridgeTransfer>, i64)> {
        let offset = (page - 1) * page_size;
        
        let transfers = sqlx::query_as::<_, BridgeTransfer>(
            r#"
            SELECT * FROM bridge_transfers 
            WHERE user_id = $1
            ORDER BY created_at DESC
            LIMIT $2 OFFSET $3
            "#
        )
        .bind(user_id)
        .bind(page_size)
        .bind(offset)
        .fetch_all(&self.pool)
        .await?;
        
        let total: (i64,) = sqlx::query_as(
            "SELECT COUNT(*) FROM bridge_transfers WHERE user_id = $1"
        )
        .bind(user_id)
        .fetch_one(&self.pool)
        .await?;
        
        Ok((transfers, total.0))
    }
    
    /// Update transfer status
    pub async fn update_transfer_status(
        &self,
        id: Uuid,
        status: TransferStatus,
        error_message: Option<&str>,
    ) -> BridgeResult<BridgeTransfer> {
        let completed_at = if status == TransferStatus::Completed {
            Some(Utc::now())
        } else {
            None
        };
        
        let transfer = sqlx::query_as::<_, BridgeTransfer>(
            r#"
            UPDATE bridge_transfers 
            SET status = $2, error_message = $3, completed_at = $4, updated_at = NOW()
            WHERE id = $1
            RETURNING *
            "#
        )
        .bind(id)
        .bind(status)
        .bind(error_message)
        .bind(completed_at)
        .fetch_one(&self.pool)
        .await?;
        
        Ok(transfer)
    }
    
    /// Update source chain confirmation
    pub async fn update_source_confirmation(
        &self,
        id: Uuid,
        tx_hash: &str,
        block_number: i64,
        confirmations: i32,
    ) -> BridgeResult<BridgeTransfer> {
        let transfer = sqlx::query_as::<_, BridgeTransfer>(
            r#"
            UPDATE bridge_transfers 
            SET 
                source_tx_hash = $2,
                source_block_number = $3,
                source_confirmations = $4,
                status = CASE 
                    WHEN status = 'pending' THEN 'source_confirmed'::transfer_status
                    ELSE status
                END,
                source_confirmed_at = COALESCE(source_confirmed_at, NOW()),
                updated_at = NOW()
            WHERE id = $1
            RETURNING *
            "#
        )
        .bind(id)
        .bind(tx_hash)
        .bind(block_number)
        .bind(confirmations)
        .fetch_one(&self.pool)
        .await?;
        
        Ok(transfer)
    }
    
    /// Update destination chain confirmation
    pub async fn update_destination_confirmation(
        &self,
        id: Uuid,
        tx_hash: &str,
        block_number: i64,
    ) -> BridgeResult<BridgeTransfer> {
        let transfer = sqlx::query_as::<_, BridgeTransfer>(
            r#"
            UPDATE bridge_transfers 
            SET 
                destination_tx_hash = $2,
                destination_block_number = $3,
                status = 'completed',
                completed_at = NOW(),
                updated_at = NOW()
            WHERE id = $1
            RETURNING *
            "#
        )
        .bind(id)
        .bind(tx_hash)
        .bind(block_number)
        .fetch_one(&self.pool)
        .await?;
        
        Ok(transfer)
    }
    
    /// Get pending transfers for chain
    pub async fn get_pending_transfers_for_chain(
        &self,
        chain: &str,
        limit: i32,
    ) -> BridgeResult<Vec<BridgeTransfer>> {
        let transfers = sqlx::query_as::<_, BridgeTransfer>(
            r#"
            SELECT * FROM bridge_transfers 
            WHERE (source_chain = $1 OR destination_chain = $1)
            AND status IN ('pending', 'source_confirmed', 'htlc_created', 'claiming')
            ORDER BY initiated_at ASC
            LIMIT $2
            "#
        )
        .bind(chain)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;
        
        Ok(transfers)
    }
    
    /// Get expired transfers
    pub async fn get_expired_transfers(&self) -> BridgeResult<Vec<BridgeTransfer>> {
        let transfers = sqlx::query_as::<_, BridgeTransfer>(
            r#"
            SELECT * FROM bridge_transfers 
            WHERE status IN ('pending', 'source_confirmed')
            AND timelock_expiry < NOW()
            "#
        )
        .fetch_all(&self.pool)
        .await?;
        
        Ok(transfers)
    }
    
    // ========================================================================
    // HTLC OPERATIONS
    // ========================================================================
    
    /// Create a new HTLC
    pub async fn create_htlc(
        &self,
        transfer_id: Uuid,
        chain: &str,
        contract_address: &str,
        sender_address: &str,
        receiver_address: &str,
        asset: &str,
        amount: Decimal,
        secret_hash: &str,
        timelock: DateTime<Utc>,
    ) -> BridgeResult<Htlc> {
        let htlc = sqlx::query_as::<_, Htlc>(
            r#"
            INSERT INTO htlcs (
                transfer_id, chain, contract_address,
                sender_address, receiver_address,
                asset, amount,
                secret_hash, timelock,
                status
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, 'pending'
            )
            RETURNING *
            "#
        )
        .bind(transfer_id)
        .bind(chain)
        .bind(contract_address)
        .bind(sender_address)
        .bind(receiver_address)
        .bind(asset)
        .bind(amount)
        .bind(secret_hash)
        .bind(timelock)
        .fetch_one(&self.pool)
        .await?;
        
        Ok(htlc)
    }
    
    /// Get HTLC by ID
    pub async fn get_htlc(&self, id: Uuid) -> BridgeResult<Option<Htlc>> {
        let htlc = sqlx::query_as::<_, Htlc>(
            "SELECT * FROM htlcs WHERE id = $1"
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;
        
        Ok(htlc)
    }
    
    /// Get HTLC by transfer ID
    pub async fn get_htlc_by_transfer(
        &self,
        transfer_id: Uuid,
        chain: &str,
    ) -> BridgeResult<Option<Htlc>> {
        let htlc = sqlx::query_as::<_, Htlc>(
            "SELECT * FROM htlcs WHERE transfer_id = $1 AND chain = $2"
        )
        .bind(transfer_id)
        .bind(chain)
        .fetch_optional(&self.pool)
        .await?;
        
        Ok(htlc)
    }
    
    /// Update HTLC as locked
    pub async fn update_htlc_locked(
        &self,
        id: Uuid,
        lock_tx_hash: &str,
        onchain_htlc_id: Option<&str>,
    ) -> BridgeResult<Htlc> {
        let htlc = sqlx::query_as::<_, Htlc>(
            r#"
            UPDATE htlcs 
            SET 
                status = 'locked',
                lock_tx_hash = $2,
                onchain_htlc_id = $3,
                locked_at = NOW(),
                updated_at = NOW()
            WHERE id = $1
            RETURNING *
            "#
        )
        .bind(id)
        .bind(lock_tx_hash)
        .bind(onchain_htlc_id)
        .fetch_one(&self.pool)
        .await?;
        
        Ok(htlc)
    }
    
    /// Update HTLC as claimed
    pub async fn update_htlc_claimed(
        &self,
        id: Uuid,
        secret: &str,
        claim_tx_hash: &str,
    ) -> BridgeResult<Htlc> {
        let htlc = sqlx::query_as::<_, Htlc>(
            r#"
            UPDATE htlcs 
            SET 
                status = 'claimed',
                secret = $2,
                claim_tx_hash = $3,
                claimed_at = NOW(),
                updated_at = NOW()
            WHERE id = $1
            RETURNING *
            "#
        )
        .bind(id)
        .bind(secret)
        .bind(claim_tx_hash)
        .fetch_one(&self.pool)
        .await?;
        
        Ok(htlc)
    }
    
    /// Update HTLC as refunded
    pub async fn update_htlc_refunded(
        &self,
        id: Uuid,
        refund_tx_hash: &str,
    ) -> BridgeResult<Htlc> {
        let htlc = sqlx::query_as::<_, Htlc>(
            r#"
            UPDATE htlcs 
            SET 
                status = 'refunded',
                refund_tx_hash = $2,
                refunded_at = NOW(),
                updated_at = NOW()
            WHERE id = $1
            RETURNING *
            "#
        )
        .bind(id)
        .bind(refund_tx_hash)
        .fetch_one(&self.pool)
        .await?;
        
        Ok(htlc)
    }
    
    /// Get expired HTLCs that can be refunded
    pub async fn get_expired_htlcs(&self) -> BridgeResult<Vec<Htlc>> {
        let htlcs = sqlx::query_as::<_, Htlc>(
            r#"
            SELECT * FROM htlcs 
            WHERE status = 'locked'
            AND timelock < NOW()
            "#
        )
        .fetch_all(&self.pool)
        .await?;
        
        Ok(htlcs)
    }
    
    // ========================================================================
    // ROUTE OPERATIONS
    // ========================================================================
    
    /// Get route for transfer
    pub async fn get_route(
        &self,
        source_chain: &str,
        destination_chain: &str,
        asset_symbol: &str,
    ) -> BridgeResult<Option<BridgeRoute>> {
        let route = sqlx::query_as::<_, BridgeRoute>(
            r#"
            SELECT * FROM bridge_routes 
            WHERE source_chain = $1 
            AND destination_chain = $2 
            AND asset_symbol = $3
            AND is_enabled = true
            "#
        )
        .bind(source_chain)
        .bind(destination_chain)
        .bind(asset_symbol)
        .fetch_optional(&self.pool)
        .await?;
        
        Ok(route)
    }
    
    /// Get all enabled routes
    pub async fn get_all_routes(&self) -> BridgeResult<Vec<BridgeRoute>> {
        let routes = sqlx::query_as::<_, BridgeRoute>(
            "SELECT * FROM bridge_routes WHERE is_enabled = true ORDER BY source_chain, destination_chain"
        )
        .fetch_all(&self.pool)
        .await?;
        
        Ok(routes)
    }
    
    /// Update route liquidity
    pub async fn update_route_liquidity(
        &self,
        id: Uuid,
        liquidity_available: Decimal,
    ) -> BridgeResult<BridgeRoute> {
        let route = sqlx::query_as::<_, BridgeRoute>(
            r#"
            UPDATE bridge_routes 
            SET liquidity_available = $2, updated_at = NOW()
            WHERE id = $1
            RETURNING *
            "#
        )
        .bind(id)
        .bind(liquidity_available)
        .fetch_one(&self.pool)
        .await?;
        
        Ok(route)
    }
    
    // ========================================================================
    // CHAIN OPERATIONS
    // ========================================================================
    
    /// Get supported chain
    pub async fn get_chain(&self, chain_id: &str) -> BridgeResult<Option<SupportedChain>> {
        let chain = sqlx::query_as::<_, SupportedChain>(
            "SELECT * FROM supported_chains WHERE chain_id = $1 AND is_enabled = true"
        )
        .bind(chain_id)
        .fetch_optional(&self.pool)
        .await?;
        
        Ok(chain)
    }
    
    /// Get all supported chains
    pub async fn get_all_chains(&self) -> BridgeResult<Vec<SupportedChain>> {
        let chains = sqlx::query_as::<_, SupportedChain>(
            "SELECT * FROM supported_chains WHERE is_enabled = true ORDER BY name"
        )
        .fetch_all(&self.pool)
        .await?;
        
        Ok(chains)
    }
    
    /// Get assets for chain
    pub async fn get_chain_assets(
        &self,
        chain_id: &str,
    ) -> BridgeResult<Vec<SupportedAsset>> {
        let assets = sqlx::query_as::<_, SupportedAsset>(
            "SELECT * FROM supported_assets WHERE chain_id = $1 AND is_enabled = true ORDER BY symbol"
        )
        .bind(chain_id)
        .fetch_all(&self.pool)
        .await?;
        
        Ok(assets)
    }
    
    // ========================================================================
    // LIQUIDITY OPERATIONS
    // ========================================================================
    
    /// Get liquidity pool
    pub async fn get_liquidity_pool(
        &self,
        chain: &str,
        asset_symbol: &str,
    ) -> BridgeResult<Option<LiquidityPool>> {
        let pool = sqlx::query_as::<_, LiquidityPool>(
            "SELECT * FROM liquidity_pools WHERE chain = $1 AND asset_symbol = $2"
        )
        .bind(chain)
        .bind(asset_symbol)
        .fetch_optional(&self.pool)
        .await?;
        
        Ok(pool)
    }
    
    /// Lock liquidity
    pub async fn lock_liquidity(
        &self,
        chain: &str,
        asset_symbol: &str,
        amount: Decimal,
    ) -> BridgeResult<LiquidityPool> {
        let pool = sqlx::query_as::<_, LiquidityPool>(
            r#"
            UPDATE liquidity_pools 
            SET 
                available_liquidity = available_liquidity - $3,
                locked_liquidity = locked_liquidity + $3,
                updated_at = NOW()
            WHERE chain = $1 AND asset_symbol = $2
            AND available_liquidity >= $3
            RETURNING *
            "#
        )
        .bind(chain)
        .bind(asset_symbol)
        .bind(amount)
        .fetch_one(&self.pool)
        .await?;
        
        Ok(pool)
    }
    
    /// Release liquidity
    pub async fn release_liquidity(
        &self,
        chain: &str,
        asset_symbol: &str,
        amount: Decimal,
    ) -> BridgeResult<LiquidityPool> {
        let pool = sqlx::query_as::<_, LiquidityPool>(
            r#"
            UPDATE liquidity_pools 
            SET 
                available_liquidity = available_liquidity + $3,
                locked_liquidity = locked_liquidity - $3,
                updated_at = NOW()
            WHERE chain = $1 AND asset_symbol = $2
            RETURNING *
            "#
        )
        .bind(chain)
        .bind(asset_symbol)
        .bind(amount)
        .fetch_one(&self.pool)
        .await?;
        
        Ok(pool)
    }
    
    // ========================================================================
    // STATISTICS
    // ========================================================================
    
    /// Get transfer statistics
    pub async fn get_transfer_stats(&self) -> BridgeResult<TransferStats> {
        let stats = sqlx::query_as::<_, TransferStats>(
            r#"
            SELECT 
                COUNT(*) as total_transfers,
                COUNT(*) FILTER (WHERE status = 'completed') as completed_transfers,
                COUNT(*) FILTER (WHERE status = 'pending') as pending_transfers,
                COUNT(*) FILTER (WHERE status = 'failed') as failed_transfers,
                COALESCE(SUM(amount_usd) FILTER (WHERE status = 'completed'), 0) as total_volume_usd,
                COALESCE(SUM(bridge_fee) FILTER (WHERE status = 'completed'), 0) as total_fees_collected
            FROM bridge_transfers
            "#
        )
        .fetch_one(&self.pool)
        .await?;
        
        Ok(stats)
    }
}

/// Transfer statistics
#[derive(Debug, Clone, sqlx::FromRow)]
pub struct TransferStats {
    pub total_transfers: i64,
    pub completed_transfers: i64,
    pub pending_transfers: i64,
    pub failed_transfers: i64,
    pub total_volume_usd: Decimal,
    pub total_fees_collected: Decimal,
}
