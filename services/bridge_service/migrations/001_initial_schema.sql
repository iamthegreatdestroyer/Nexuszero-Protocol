-- Bridge Service Initial Schema
-- NexusZero Protocol - Phase 1
-- Cross-Chain Bridge with Atomic Swaps (HTLC)

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================================================
-- SUPPORTED CHAINS
-- ============================================================================

CREATE TABLE supported_chains (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chain_id VARCHAR(50) NOT NULL UNIQUE,
    name VARCHAR(100) NOT NULL,
    chain_type VARCHAR(50) NOT NULL,  -- evm, bitcoin, solana, cosmos, substrate
    native_token VARCHAR(20) NOT NULL,
    
    -- Network parameters
    confirmations_required INTEGER NOT NULL DEFAULT 12,
    block_time_secs INTEGER NOT NULL DEFAULT 12,
    
    -- Contract addresses
    bridge_contract VARCHAR(255),
    htlc_contract VARCHAR(255),
    
    -- Configuration
    config JSONB NOT NULL DEFAULT '{}',
    
    -- Status
    is_enabled BOOLEAN NOT NULL DEFAULT true,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_chains_chain_id ON supported_chains(chain_id);
CREATE INDEX idx_chains_enabled ON supported_chains(is_enabled) WHERE is_enabled = true;

-- ============================================================================
-- SUPPORTED ASSETS
-- ============================================================================

CREATE TABLE supported_assets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chain_id VARCHAR(50) NOT NULL REFERENCES supported_chains(chain_id),
    
    -- Asset identification
    symbol VARCHAR(20) NOT NULL,
    name VARCHAR(100) NOT NULL,
    contract_address VARCHAR(255),  -- NULL for native tokens
    decimals SMALLINT NOT NULL DEFAULT 18,
    
    -- Limits
    min_amount DECIMAL(38, 18) NOT NULL DEFAULT 0,
    max_amount DECIMAL(38, 18) NOT NULL DEFAULT 1000000000,
    
    -- Status
    is_enabled BOOLEAN NOT NULL DEFAULT true,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    UNIQUE(chain_id, symbol)
);

CREATE INDEX idx_assets_chain ON supported_assets(chain_id);
CREATE INDEX idx_assets_symbol ON supported_assets(symbol);

-- ============================================================================
-- BRIDGE ROUTES
-- ============================================================================

CREATE TABLE bridge_routes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Route definition
    source_chain VARCHAR(50) NOT NULL,
    destination_chain VARCHAR(50) NOT NULL,
    asset_symbol VARCHAR(20) NOT NULL,
    
    -- Route configuration
    is_enabled BOOLEAN NOT NULL DEFAULT true,
    fee_bps INTEGER NOT NULL DEFAULT 30,  -- 0.30%
    
    -- Limits
    min_amount DECIMAL(38, 18) NOT NULL DEFAULT 0,
    max_amount DECIMAL(38, 18) NOT NULL DEFAULT 1000000,
    
    -- Estimates
    estimated_time_secs INTEGER NOT NULL DEFAULT 300,
    
    -- Liquidity
    liquidity_available DECIMAL(38, 18) NOT NULL DEFAULT 0,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    UNIQUE(source_chain, destination_chain, asset_symbol)
);

CREATE INDEX idx_routes_source ON bridge_routes(source_chain);
CREATE INDEX idx_routes_destination ON bridge_routes(destination_chain);
CREATE INDEX idx_routes_enabled ON bridge_routes(is_enabled) WHERE is_enabled = true;

-- ============================================================================
-- BRIDGE TRANSFERS
-- ============================================================================

CREATE TYPE transfer_status AS ENUM (
    'pending',
    'source_confirmed',
    'htlc_created',
    'htlc_matched',
    'claiming',
    'completed',
    'refunded',
    'failed',
    'cancelled',
    'expired'
);

CREATE TYPE transfer_direction AS ENUM (
    'lock',
    'burn',
    'atomic_swap'
);

CREATE TABLE bridge_transfers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Transfer identification
    transfer_id VARCHAR(50) NOT NULL UNIQUE,
    user_id UUID NOT NULL,
    
    -- Source chain info
    source_chain VARCHAR(50) NOT NULL,
    source_address VARCHAR(255) NOT NULL,
    source_tx_hash VARCHAR(255),
    source_block_number BIGINT,
    source_confirmations INTEGER NOT NULL DEFAULT 0,
    
    -- Destination chain info
    destination_chain VARCHAR(50) NOT NULL,
    destination_address VARCHAR(255) NOT NULL,
    destination_tx_hash VARCHAR(255),
    destination_block_number BIGINT,
    
    -- Asset and amount
    asset_symbol VARCHAR(20) NOT NULL,
    asset_address VARCHAR(255),
    amount DECIMAL(38, 18) NOT NULL,
    amount_usd DECIMAL(20, 2),
    
    -- Fees
    bridge_fee DECIMAL(38, 18) NOT NULL DEFAULT 0,
    gas_fee_source DECIMAL(38, 18),
    gas_fee_destination DECIMAL(38, 18),
    
    -- HTLC details
    htlc_id UUID,
    secret_hash VARCHAR(64),
    timelock_expiry TIMESTAMP WITH TIME ZONE,
    
    -- Status
    status transfer_status NOT NULL DEFAULT 'pending',
    direction transfer_direction NOT NULL DEFAULT 'lock',
    error_message TEXT,
    retry_count INTEGER NOT NULL DEFAULT 0,
    
    -- Timestamps
    initiated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    source_confirmed_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_transfers_transfer_id ON bridge_transfers(transfer_id);
CREATE INDEX idx_transfers_user ON bridge_transfers(user_id);
CREATE INDEX idx_transfers_status ON bridge_transfers(status);
CREATE INDEX idx_transfers_source_chain ON bridge_transfers(source_chain);
CREATE INDEX idx_transfers_destination_chain ON bridge_transfers(destination_chain);
CREATE INDEX idx_transfers_created ON bridge_transfers(created_at DESC);
CREATE INDEX idx_transfers_pending ON bridge_transfers(status, source_chain) 
    WHERE status IN ('pending', 'source_confirmed', 'htlc_created', 'claiming');

-- ============================================================================
-- HTLC (Hash Time-Locked Contracts)
-- ============================================================================

CREATE TYPE htlc_status AS ENUM (
    'pending',
    'locked',
    'claimed',
    'refunded',
    'expired'
);

CREATE TABLE htlcs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Associated transfer
    transfer_id UUID NOT NULL REFERENCES bridge_transfers(id),
    
    -- Chain info
    chain VARCHAR(50) NOT NULL,
    contract_address VARCHAR(255) NOT NULL,
    onchain_htlc_id VARCHAR(255),
    
    -- Participants
    sender_address VARCHAR(255) NOT NULL,
    receiver_address VARCHAR(255) NOT NULL,
    
    -- Asset and amount
    asset VARCHAR(20) NOT NULL,
    amount DECIMAL(38, 18) NOT NULL,
    
    -- Hashlock
    secret_hash VARCHAR(64) NOT NULL,
    secret VARCHAR(64),
    
    -- Timelock
    timelock TIMESTAMP WITH TIME ZONE NOT NULL,
    timelock_block BIGINT,
    
    -- Transaction hashes
    lock_tx_hash VARCHAR(255),
    claim_tx_hash VARCHAR(255),
    refund_tx_hash VARCHAR(255),
    
    -- Status
    status htlc_status NOT NULL DEFAULT 'pending',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    locked_at TIMESTAMP WITH TIME ZONE,
    claimed_at TIMESTAMP WITH TIME ZONE,
    refunded_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_htlcs_transfer ON htlcs(transfer_id);
CREATE INDEX idx_htlcs_chain ON htlcs(chain);
CREATE INDEX idx_htlcs_status ON htlcs(status);
CREATE INDEX idx_htlcs_timelock ON htlcs(timelock) WHERE status = 'locked';
CREATE INDEX idx_htlcs_secret_hash ON htlcs(secret_hash);

-- ============================================================================
-- LIQUIDITY POOLS
-- ============================================================================

CREATE TABLE liquidity_pools (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Pool identification
    chain VARCHAR(50) NOT NULL,
    asset_symbol VARCHAR(20) NOT NULL,
    
    -- Liquidity amounts
    total_liquidity DECIMAL(38, 18) NOT NULL DEFAULT 0,
    available_liquidity DECIMAL(38, 18) NOT NULL DEFAULT 0,
    locked_liquidity DECIMAL(38, 18) NOT NULL DEFAULT 0,
    
    -- Metrics
    utilization_rate DECIMAL(10, 8) NOT NULL DEFAULT 0,
    apy DECIMAL(10, 4) NOT NULL DEFAULT 0,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    UNIQUE(chain, asset_symbol)
);

CREATE INDEX idx_pools_chain_asset ON liquidity_pools(chain, asset_symbol);

-- ============================================================================
-- LIQUIDITY POSITIONS
-- ============================================================================

CREATE TABLE liquidity_positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- User and pool
    user_id UUID NOT NULL,
    pool_id UUID NOT NULL REFERENCES liquidity_pools(id),
    
    -- Position details
    chain VARCHAR(50) NOT NULL,
    asset_symbol VARCHAR(20) NOT NULL,
    amount DECIMAL(38, 18) NOT NULL,
    share_percentage DECIMAL(10, 8) NOT NULL DEFAULT 0,
    
    -- Rewards
    rewards_earned DECIMAL(38, 18) NOT NULL DEFAULT 0,
    rewards_claimed DECIMAL(38, 18) NOT NULL DEFAULT 0,
    
    -- Timestamps
    deposited_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_positions_user ON liquidity_positions(user_id);
CREATE INDEX idx_positions_pool ON liquidity_positions(pool_id);
CREATE UNIQUE INDEX idx_positions_user_pool ON liquidity_positions(user_id, pool_id);

-- ============================================================================
-- RELAYERS
-- ============================================================================

CREATE TYPE relayer_status AS ENUM (
    'active',
    'paused',
    'deactivated'
);

CREATE TABLE relayers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Relayer identification
    address VARCHAR(255) NOT NULL UNIQUE,
    chains VARCHAR(50)[] NOT NULL DEFAULT '{}',
    
    -- Status
    status relayer_status NOT NULL DEFAULT 'active',
    
    -- Stake
    stake_amount DECIMAL(38, 18) NOT NULL DEFAULT 0,
    
    -- Performance metrics
    successful_relays BIGINT NOT NULL DEFAULT 0,
    failed_relays BIGINT NOT NULL DEFAULT 0,
    total_volume_usd DECIMAL(20, 2) NOT NULL DEFAULT 0,
    reputation_score DECIMAL(5, 4) NOT NULL DEFAULT 1.0000,
    
    -- Activity
    last_active_at TIMESTAMP WITH TIME ZONE,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_relayers_status ON relayers(status);
CREATE INDEX idx_relayers_reputation ON relayers(reputation_score DESC);

-- ============================================================================
-- RELAY TASKS
-- ============================================================================

CREATE TABLE relay_tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Associated transfer
    transfer_id UUID NOT NULL REFERENCES bridge_transfers(id),
    
    -- Assigned relayer
    relayer_id UUID REFERENCES relayers(id),
    
    -- Task details
    task_type VARCHAR(50) NOT NULL,  -- lock, claim, refund
    chain VARCHAR(50) NOT NULL,
    
    -- Execution
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    tx_hash VARCHAR(255),
    gas_used DECIMAL(38, 18),
    error_message TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    
    -- Scheduling
    scheduled_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_relay_tasks_transfer ON relay_tasks(transfer_id);
CREATE INDEX idx_relay_tasks_relayer ON relay_tasks(relayer_id);
CREATE INDEX idx_relay_tasks_status ON relay_tasks(status);
CREATE INDEX idx_relay_tasks_pending ON relay_tasks(scheduled_at) WHERE status = 'pending';

-- ============================================================================
-- QUOTES
-- ============================================================================

CREATE TABLE quotes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Quote identification
    quote_id VARCHAR(50) NOT NULL UNIQUE,
    
    -- Route
    source_chain VARCHAR(50) NOT NULL,
    destination_chain VARCHAR(50) NOT NULL,
    asset_symbol VARCHAR(20) NOT NULL,
    
    -- Amounts
    input_amount DECIMAL(38, 18) NOT NULL,
    output_amount DECIMAL(38, 18) NOT NULL,
    
    -- Fees
    bridge_fee DECIMAL(38, 18) NOT NULL,
    gas_fee_estimate DECIMAL(38, 18) NOT NULL,
    
    -- Validity
    valid_until TIMESTAMP WITH TIME ZONE NOT NULL,
    executed BOOLEAN NOT NULL DEFAULT false,
    
    -- Associated transfer (if executed)
    transfer_id UUID REFERENCES bridge_transfers(id),
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_quotes_quote_id ON quotes(quote_id);
CREATE INDEX idx_quotes_valid ON quotes(valid_until) WHERE NOT executed;

-- ============================================================================
-- BRIDGE EVENTS (Audit Log)
-- ============================================================================

CREATE TABLE bridge_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Event identification
    event_type VARCHAR(100) NOT NULL,
    
    -- References
    transfer_id UUID REFERENCES bridge_transfers(id),
    htlc_id UUID REFERENCES htlcs(id),
    
    -- Chain info
    chain VARCHAR(50),
    tx_hash VARCHAR(255),
    block_number BIGINT,
    
    -- Event data
    data JSONB NOT NULL DEFAULT '{}',
    
    -- Timestamp
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_events_type ON bridge_events(event_type);
CREATE INDEX idx_events_transfer ON bridge_events(transfer_id);
CREATE INDEX idx_events_chain ON bridge_events(chain);
CREATE INDEX idx_events_created ON bridge_events(created_at DESC);

-- ============================================================================
-- FUNCTIONS AND TRIGGERS
-- ============================================================================

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply triggers
CREATE TRIGGER update_chains_updated_at
    BEFORE UPDATE ON supported_chains
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_assets_updated_at
    BEFORE UPDATE ON supported_assets
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_routes_updated_at
    BEFORE UPDATE ON bridge_routes
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_transfers_updated_at
    BEFORE UPDATE ON bridge_transfers
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_htlcs_updated_at
    BEFORE UPDATE ON htlcs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_pools_updated_at
    BEFORE UPDATE ON liquidity_pools
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_positions_updated_at
    BEFORE UPDATE ON liquidity_positions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_relayers_updated_at
    BEFORE UPDATE ON relayers
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- SEED DATA
-- ============================================================================

-- Insert supported chains
INSERT INTO supported_chains (chain_id, name, chain_type, native_token, confirmations_required, block_time_secs) VALUES
    ('ethereum', 'Ethereum', 'evm', 'ETH', 12, 12),
    ('polygon', 'Polygon', 'evm', 'MATIC', 128, 2),
    ('arbitrum', 'Arbitrum One', 'evm', 'ETH', 64, 1),
    ('optimism', 'Optimism', 'evm', 'ETH', 64, 2),
    ('bsc', 'BNB Smart Chain', 'evm', 'BNB', 15, 3),
    ('avalanche', 'Avalanche C-Chain', 'evm', 'AVAX', 12, 2),
    ('base', 'Base', 'evm', 'ETH', 64, 2),
    ('solana', 'Solana', 'solana', 'SOL', 32, 0);

-- Insert supported assets
INSERT INTO supported_assets (chain_id, symbol, name, decimals) VALUES
    ('ethereum', 'ETH', 'Ethereum', 18),
    ('ethereum', 'USDC', 'USD Coin', 6),
    ('ethereum', 'USDT', 'Tether USD', 6),
    ('ethereum', 'WBTC', 'Wrapped Bitcoin', 8),
    ('polygon', 'MATIC', 'Polygon', 18),
    ('polygon', 'USDC', 'USD Coin', 6),
    ('polygon', 'USDT', 'Tether USD', 6),
    ('arbitrum', 'ETH', 'Ethereum', 18),
    ('arbitrum', 'USDC', 'USD Coin', 6),
    ('optimism', 'ETH', 'Ethereum', 18),
    ('optimism', 'USDC', 'USD Coin', 6),
    ('bsc', 'BNB', 'BNB', 18),
    ('bsc', 'USDT', 'Tether USD', 18),
    ('avalanche', 'AVAX', 'Avalanche', 18),
    ('avalanche', 'USDC', 'USD Coin', 6),
    ('base', 'ETH', 'Ethereum', 18),
    ('base', 'USDC', 'USD Coin', 6),
    ('solana', 'SOL', 'Solana', 9),
    ('solana', 'USDC', 'USD Coin', 6);

-- Insert bridge routes (subset of common routes)
INSERT INTO bridge_routes (source_chain, destination_chain, asset_symbol, fee_bps, estimated_time_secs, liquidity_available) VALUES
    ('ethereum', 'polygon', 'USDC', 30, 300, 1000000),
    ('ethereum', 'arbitrum', 'USDC', 25, 180, 2000000),
    ('ethereum', 'optimism', 'USDC', 25, 180, 1500000),
    ('ethereum', 'base', 'USDC', 25, 180, 1000000),
    ('polygon', 'ethereum', 'USDC', 30, 600, 800000),
    ('arbitrum', 'ethereum', 'USDC', 25, 420, 1500000),
    ('arbitrum', 'optimism', 'USDC', 20, 120, 500000),
    ('optimism', 'arbitrum', 'USDC', 20, 120, 500000),
    ('ethereum', 'polygon', 'ETH', 30, 300, 500),
    ('ethereum', 'arbitrum', 'ETH', 25, 180, 1000),
    ('polygon', 'ethereum', 'MATIC', 35, 600, 100000);

-- Insert liquidity pools
INSERT INTO liquidity_pools (chain, asset_symbol, total_liquidity, available_liquidity, apy) VALUES
    ('ethereum', 'USDC', 5000000, 4500000, 5.5),
    ('ethereum', 'ETH', 2000, 1800, 4.2),
    ('polygon', 'USDC', 2000000, 1800000, 6.0),
    ('polygon', 'MATIC', 500000, 450000, 8.0),
    ('arbitrum', 'USDC', 3000000, 2700000, 5.0),
    ('arbitrum', 'ETH', 1500, 1350, 4.0),
    ('optimism', 'USDC', 2000000, 1800000, 5.5),
    ('base', 'USDC', 1000000, 900000, 6.5);

COMMENT ON TABLE supported_chains IS 'Blockchain networks supported by the bridge';
COMMENT ON TABLE supported_assets IS 'Assets that can be bridged on each chain';
COMMENT ON TABLE bridge_routes IS 'Available bridging routes between chains';
COMMENT ON TABLE bridge_transfers IS 'Individual cross-chain transfer records';
COMMENT ON TABLE htlcs IS 'Hash Time-Locked Contracts for atomic swaps';
COMMENT ON TABLE liquidity_pools IS 'Liquidity pools for each chain/asset pair';
COMMENT ON TABLE liquidity_positions IS 'Liquidity provider positions';
COMMENT ON TABLE relayers IS 'Registered bridge relayers';
COMMENT ON TABLE relay_tasks IS 'Tasks assigned to relayers';
COMMENT ON TABLE quotes IS 'Bridge transfer quotes';
COMMENT ON TABLE bridge_events IS 'Audit log of bridge events';
