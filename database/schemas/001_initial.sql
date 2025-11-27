-- ============================================================
-- NexusZero Protocol Database Schema
-- Version: 001_initial
-- Description: Core tables for privacy-preserving transactions
-- ============================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================
-- ENUM TYPES
-- ============================================================

CREATE TYPE privacy_level AS ENUM (
    'transparent',      -- Level 0
    'pseudonymous',     -- Level 1
    'confidential',     -- Level 2
    'private',          -- Level 3
    'anonymous',        -- Level 4
    'sovereign'         -- Level 5
);

CREATE TYPE transaction_status AS ENUM (
    'created',
    'pending',
    'proving',
    'submitted',
    'confirmed',
    'failed',
    'cancelled'
);

CREATE TYPE proof_type AS ENUM (
    'transaction',
    'compliance',
    'bridge',
    'identity',
    'range',
    'membership'
);

CREATE TYPE node_status AS ENUM (
    'active',
    'inactive',
    'suspended',
    'maintenance'
);

CREATE TYPE bridge_status AS ENUM (
    'initiated',
    'source_locked',
    'proof_generated',
    'target_minting',
    'completed',
    'failed',
    'refunded'
);

-- ============================================================
-- USERS TABLE
-- ============================================================

CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    external_id VARCHAR(255) UNIQUE,
    public_key BYTEA NOT NULL,
    public_key_type VARCHAR(50) NOT NULL DEFAULT 'ed25519',
    encrypted_private_key BYTEA,
    key_derivation_params JSONB,
    privacy_preferences JSONB DEFAULT '{
        "default_level": 3,
        "auto_morph": true,
        "compliance_enabled": false,
        "preferred_chains": ["ethereum", "polygon"]
    }',
    kyc_status VARCHAR(50) DEFAULT 'none',
    kyc_level INTEGER DEFAULT 0,
    jurisdiction VARCHAR(10),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_active_at TIMESTAMPTZ,
    
    CONSTRAINT valid_kyc_level CHECK (kyc_level >= 0 AND kyc_level <= 3)
);

CREATE INDEX idx_users_external_id ON users(external_id);
CREATE INDEX idx_users_public_key ON users USING hash(public_key);
CREATE INDEX idx_users_created_at ON users(created_at);
CREATE INDEX idx_users_jurisdiction ON users(jurisdiction) WHERE jurisdiction IS NOT NULL;

-- ============================================================
-- TRANSACTIONS TABLE
-- ============================================================

CREATE TABLE transactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    
    -- Cryptographic commitments (hidden values)
    sender_commitment BYTEA NOT NULL,
    recipient_commitment BYTEA NOT NULL,
    amount_commitment BYTEA,
    nonce_commitment BYTEA,
    
    -- Privacy parameters
    privacy_level SMALLINT NOT NULL CHECK (privacy_level >= 0 AND privacy_level <= 5),
    privacy_level_enum privacy_level,
    morphing_enabled BOOLEAN DEFAULT false,
    morphing_schedule JSONB,
    
    -- Proof reference
    proof_id UUID,
    
    -- Chain information
    chain VARCHAR(50) NOT NULL,
    chain_tx_hash BYTEA,
    block_number BIGINT,
    gas_used BIGINT,
    gas_price NUMERIC(78, 0),
    
    -- Status and metadata
    status transaction_status NOT NULL DEFAULT 'created',
    error_message TEXT,
    metadata JSONB DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    submitted_at TIMESTAMPTZ,
    confirmed_at TIMESTAMPTZ,
    
    -- Indexes for common queries
    CONSTRAINT valid_chain CHECK (chain IN (
        'ethereum', 'polygon', 'arbitrum', 'optimism', 'base',
        'bitcoin', 'solana', 'cosmos', 'avalanche', 'bsc'
    ))
);

CREATE INDEX idx_transactions_user_id ON transactions(user_id);
CREATE INDEX idx_transactions_status ON transactions(status);
CREATE INDEX idx_transactions_chain ON transactions(chain);
CREATE INDEX idx_transactions_privacy_level ON transactions(privacy_level);
CREATE INDEX idx_transactions_created_at ON transactions(created_at DESC);
CREATE INDEX idx_transactions_chain_tx_hash ON transactions(chain_tx_hash) WHERE chain_tx_hash IS NOT NULL;
CREATE INDEX idx_transactions_pending ON transactions(created_at) WHERE status IN ('created', 'pending', 'proving');

-- ============================================================
-- PROOFS TABLE
-- ============================================================

CREATE TABLE proofs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    transaction_id UUID REFERENCES transactions(id) ON DELETE SET NULL,
    
    -- Proof data
    proof_data BYTEA NOT NULL,
    proof_hash BYTEA NOT NULL,
    proof_type proof_type NOT NULL,
    proof_system VARCHAR(50) NOT NULL DEFAULT 'groth16',
    
    -- Privacy and security
    privacy_level SMALLINT NOT NULL CHECK (privacy_level >= 0 AND privacy_level <= 5),
    security_bits INTEGER NOT NULL,
    
    -- Generation metadata
    prover_node_id UUID,
    generation_time_ms INTEGER,
    proof_size_bytes INTEGER,
    circuit_id VARCHAR(100),
    
    -- Verification
    verified BOOLEAN DEFAULT FALSE,
    verification_time TIMESTAMPTZ,
    verifier_address BYTEA,
    on_chain_verified BOOLEAN DEFAULT FALSE,
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    
    CONSTRAINT valid_proof_system CHECK (proof_system IN (
        'groth16', 'plonk', 'stark', 'bulletproofs', 'halo2', 'nova'
    ))
);

CREATE INDEX idx_proofs_transaction_id ON proofs(transaction_id);
CREATE INDEX idx_proofs_proof_type ON proofs(proof_type);
CREATE INDEX idx_proofs_prover_node_id ON proofs(prover_node_id);
CREATE INDEX idx_proofs_verified ON proofs(verified);
CREATE INDEX idx_proofs_created_at ON proofs(created_at DESC);
CREATE INDEX idx_proofs_proof_hash ON proofs USING hash(proof_hash);

-- Add foreign key from transactions to proofs
ALTER TABLE transactions ADD CONSTRAINT fk_transactions_proof
    FOREIGN KEY (proof_id) REFERENCES proofs(id) ON DELETE SET NULL;

-- ============================================================
-- COMPLIANCE RECORDS TABLE
-- ============================================================

CREATE TABLE compliance_records (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    
    -- Compliance proof details
    proof_type VARCHAR(100) NOT NULL,
    proof_data BYTEA NOT NULL,
    proof_hash BYTEA NOT NULL,
    
    -- Verification status
    verified BOOLEAN DEFAULT FALSE,
    verifier_id UUID,
    verification_method VARCHAR(100),
    
    -- Validity period
    issued_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL,
    revoked_at TIMESTAMPTZ,
    revocation_reason TEXT,
    
    -- Jurisdiction and requirements
    jurisdiction VARCHAR(10),
    regulation_type VARCHAR(100),
    required_level INTEGER DEFAULT 1,
    
    -- Selective disclosure
    disclosed_attributes JSONB DEFAULT '{}',
    hidden_attributes TEXT[],
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    CONSTRAINT valid_required_level CHECK (required_level >= 1 AND required_level <= 5),
    CONSTRAINT valid_expiry CHECK (expires_at > issued_at)
);

CREATE INDEX idx_compliance_user_id ON compliance_records(user_id);
CREATE INDEX idx_compliance_proof_type ON compliance_records(proof_type);
CREATE INDEX idx_compliance_expires_at ON compliance_records(expires_at);
CREATE INDEX idx_compliance_jurisdiction ON compliance_records(jurisdiction);
CREATE INDEX idx_compliance_active ON compliance_records(user_id, expires_at) 
    WHERE revoked_at IS NULL AND verified = TRUE;

-- ============================================================
-- BRIDGE TRANSFERS TABLE
-- ============================================================

CREATE TABLE bridge_transfers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    
    -- Chain information
    source_chain VARCHAR(50) NOT NULL,
    target_chain VARCHAR(50) NOT NULL,
    
    -- Transaction hashes
    source_tx_hash BYTEA,
    target_tx_hash BYTEA,
    
    -- Amount (hidden via commitment)
    amount_commitment BYTEA NOT NULL,
    
    -- Privacy and proof
    privacy_level SMALLINT NOT NULL CHECK (privacy_level >= 0 AND privacy_level <= 5),
    proof_id UUID REFERENCES proofs(id),
    
    -- Relay and verification
    relay_node_id UUID,
    attestations JSONB DEFAULT '[]',
    required_attestations INTEGER DEFAULT 3,
    
    -- Status tracking
    status bridge_status NOT NULL DEFAULT 'initiated',
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    
    -- Fees
    bridge_fee_commitment BYTEA,
    gas_fee_source NUMERIC(78, 0),
    gas_fee_target NUMERIC(78, 0),
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    source_confirmed_at TIMESTAMPTZ,
    target_confirmed_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    
    CONSTRAINT different_chains CHECK (source_chain != target_chain)
);

CREATE INDEX idx_bridge_user_id ON bridge_transfers(user_id);
CREATE INDEX idx_bridge_status ON bridge_transfers(status);
CREATE INDEX idx_bridge_source_chain ON bridge_transfers(source_chain);
CREATE INDEX idx_bridge_target_chain ON bridge_transfers(target_chain);
CREATE INDEX idx_bridge_created_at ON bridge_transfers(created_at DESC);
CREATE INDEX idx_bridge_pending ON bridge_transfers(created_at) 
    WHERE status NOT IN ('completed', 'failed', 'refunded');

-- ============================================================
-- PROVER NODES TABLE
-- ============================================================

CREATE TABLE prover_nodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Identity
    public_key BYTEA NOT NULL UNIQUE,
    node_name VARCHAR(255),
    operator_address BYTEA,
    
    -- Rewards
    reward_address VARCHAR(255) NOT NULL,
    total_rewards_earned NUMERIC(78, 0) DEFAULT 0,
    pending_rewards NUMERIC(78, 0) DEFAULT 0,
    
    -- Capabilities
    supported_levels SMALLINT[] NOT NULL,
    supported_proof_systems TEXT[] DEFAULT ARRAY['groth16'],
    gpu_enabled BOOLEAN DEFAULT FALSE,
    gpu_model VARCHAR(100),
    max_concurrent_proofs INTEGER DEFAULT 4,
    
    -- Performance metrics
    reputation_score DECIMAL(5,2) DEFAULT 100.00,
    total_proofs_generated BIGINT DEFAULT 0,
    total_proofs_failed BIGINT DEFAULT 0,
    average_generation_time_ms INTEGER,
    uptime_percentage DECIMAL(5,2) DEFAULT 100.00,
    
    -- Current state
    status node_status NOT NULL DEFAULT 'active',
    current_load INTEGER DEFAULT 0,
    last_heartbeat TIMESTAMPTZ,
    last_proof_at TIMESTAMPTZ,
    
    -- Network information
    endpoint_url VARCHAR(500),
    region VARCHAR(50),
    version VARCHAR(50),
    
    -- Staking
    staked_amount NUMERIC(78, 0) DEFAULT 0,
    slashed_amount NUMERIC(78, 0) DEFAULT 0,
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    CONSTRAINT valid_reputation CHECK (reputation_score >= 0 AND reputation_score <= 100),
    CONSTRAINT valid_uptime CHECK (uptime_percentage >= 0 AND uptime_percentage <= 100)
);

CREATE INDEX idx_prover_nodes_status ON prover_nodes(status);
CREATE INDEX idx_prover_nodes_reputation ON prover_nodes(reputation_score DESC);
CREATE INDEX idx_prover_nodes_supported_levels ON prover_nodes USING gin(supported_levels);
CREATE INDEX idx_prover_nodes_gpu ON prover_nodes(gpu_enabled) WHERE gpu_enabled = TRUE;
CREATE INDEX idx_prover_nodes_active ON prover_nodes(reputation_score DESC, current_load) 
    WHERE status = 'active';
CREATE INDEX idx_prover_nodes_region ON prover_nodes(region);

-- ============================================================
-- AUDIT LOG TABLE
-- ============================================================

CREATE TABLE audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Action details
    action VARCHAR(100) NOT NULL,
    entity_type VARCHAR(50) NOT NULL,
    entity_id UUID,
    
    -- Actor information
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    ip_address INET,
    user_agent TEXT,
    
    -- Change data
    old_values JSONB,
    new_values JSONB,
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_audit_log_action ON audit_log(action);
CREATE INDEX idx_audit_log_entity ON audit_log(entity_type, entity_id);
CREATE INDEX idx_audit_log_user_id ON audit_log(user_id);
CREATE INDEX idx_audit_log_created_at ON audit_log(created_at DESC);

-- ============================================================
-- PRIVACY MORPHING SCHEDULES TABLE
-- ============================================================

CREATE TABLE morphing_schedules (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    transaction_id UUID REFERENCES transactions(id) ON DELETE CASCADE,
    
    -- Morphing configuration
    current_level SMALLINT NOT NULL,
    target_level SMALLINT NOT NULL,
    morph_direction VARCHAR(10) NOT NULL, -- 'increase' or 'decrease'
    
    -- Schedule
    schedule_type VARCHAR(50) NOT NULL, -- 'time_based', 'block_based', 'event_based'
    schedule_params JSONB NOT NULL,
    next_morph_at TIMESTAMPTZ,
    
    -- Status
    completed BOOLEAN DEFAULT FALSE,
    morphs_completed INTEGER DEFAULT 0,
    total_morphs INTEGER,
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_morphed_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    
    CONSTRAINT valid_levels CHECK (
        current_level >= 0 AND current_level <= 5 AND
        target_level >= 0 AND target_level <= 5 AND
        current_level != target_level
    )
);

CREATE INDEX idx_morphing_transaction_id ON morphing_schedules(transaction_id);
CREATE INDEX idx_morphing_next_morph ON morphing_schedules(next_morph_at) 
    WHERE completed = FALSE;

-- ============================================================
-- FUNCTIONS AND TRIGGERS
-- ============================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply updated_at triggers
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_transactions_updated_at
    BEFORE UPDATE ON transactions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_prover_nodes_updated_at
    BEFORE UPDATE ON prover_nodes
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Function to log changes to audit table
CREATE OR REPLACE FUNCTION audit_log_changes()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO audit_log (action, entity_type, entity_id, new_values)
        VALUES ('CREATE', TG_TABLE_NAME, NEW.id, to_jsonb(NEW));
        RETURN NEW;
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO audit_log (action, entity_type, entity_id, old_values, new_values)
        VALUES ('UPDATE', TG_TABLE_NAME, NEW.id, to_jsonb(OLD), to_jsonb(NEW));
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO audit_log (action, entity_type, entity_id, old_values)
        VALUES ('DELETE', TG_TABLE_NAME, OLD.id, to_jsonb(OLD));
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$$ language 'plpgsql';

-- Apply audit triggers to sensitive tables
CREATE TRIGGER audit_transactions
    AFTER INSERT OR UPDATE OR DELETE ON transactions
    FOR EACH ROW
    EXECUTE FUNCTION audit_log_changes();

CREATE TRIGGER audit_compliance_records
    AFTER INSERT OR UPDATE OR DELETE ON compliance_records
    FOR EACH ROW
    EXECUTE FUNCTION audit_log_changes();

CREATE TRIGGER audit_bridge_transfers
    AFTER INSERT OR UPDATE OR DELETE ON bridge_transfers
    FOR EACH ROW
    EXECUTE FUNCTION audit_log_changes();

-- ============================================================
-- VIEWS
-- ============================================================

-- View for active prover nodes
CREATE VIEW active_prover_nodes AS
SELECT 
    id,
    node_name,
    supported_levels,
    gpu_enabled,
    reputation_score,
    current_load,
    region,
    last_heartbeat
FROM prover_nodes
WHERE status = 'active'
    AND last_heartbeat > NOW() - INTERVAL '5 minutes'
ORDER BY reputation_score DESC, current_load ASC;

-- View for transaction statistics
CREATE VIEW transaction_stats AS
SELECT 
    chain,
    privacy_level,
    status,
    COUNT(*) as count,
    DATE_TRUNC('day', created_at) as date
FROM transactions
GROUP BY chain, privacy_level, status, DATE_TRUNC('day', created_at);

-- View for pending bridge transfers requiring attention
CREATE VIEW pending_bridge_transfers AS
SELECT 
    bt.*,
    u.external_id as user_external_id
FROM bridge_transfers bt
LEFT JOIN users u ON bt.user_id = u.id
WHERE bt.status NOT IN ('completed', 'failed', 'refunded')
    AND bt.created_at < NOW() - INTERVAL '10 minutes'
ORDER BY bt.created_at ASC;

-- ============================================================
-- COMMENTS
-- ============================================================

COMMENT ON TABLE users IS 'User accounts with public keys and privacy preferences';
COMMENT ON TABLE transactions IS 'Privacy-preserving transactions with cryptographic commitments';
COMMENT ON TABLE proofs IS 'Zero-knowledge proofs for transactions and compliance';
COMMENT ON TABLE compliance_records IS 'Regulatory compliance proofs with selective disclosure';
COMMENT ON TABLE bridge_transfers IS 'Cross-chain bridge transfers with privacy preservation';
COMMENT ON TABLE prover_nodes IS 'Distributed prover network nodes';
COMMENT ON TABLE audit_log IS 'Immutable audit log of all system changes';
COMMENT ON TABLE morphing_schedules IS 'Privacy level morphing schedules for dynamic privacy';

-- ============================================================
-- GRANTS (adjust based on your application roles)
-- ============================================================

-- Create application role
-- CREATE ROLE nexuszero_app WITH LOGIN PASSWORD 'secure_password';

-- Grant permissions
-- GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO nexuszero_app;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO nexuszero_app;
-- GRANT SELECT ON ALL TABLES IN SCHEMA public TO nexuszero_readonly;
