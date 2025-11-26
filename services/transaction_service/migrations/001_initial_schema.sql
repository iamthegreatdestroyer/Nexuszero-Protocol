-- NexusZero Transaction Service - Initial Schema
-- Migration: 001_initial_schema

-- Create custom types
CREATE TYPE transaction_status AS ENUM (
    'pending',
    'proof_generating',
    'proof_ready',
    'submitted',
    'confirmed',
    'finalized',
    'failed',
    'cancelled'
);

-- Create transactions table
CREATE TABLE IF NOT EXISTS transactions (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL,
    sender VARCHAR(256) NOT NULL,
    recipient VARCHAR(256) NOT NULL,
    amount BIGINT NOT NULL,
    asset_id VARCHAR(128) NOT NULL,
    privacy_level SMALLINT NOT NULL DEFAULT 4 CHECK (privacy_level >= 0 AND privacy_level <= 5),
    status transaction_status NOT NULL DEFAULT 'pending',
    chain_id VARCHAR(64) NOT NULL,
    chain_tx_hash VARCHAR(256),
    proof TEXT,
    proof_id UUID,
    memo TEXT,
    metadata JSONB,
    error_message TEXT,
    block_number BIGINT,
    gas_used BIGINT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finalized_at TIMESTAMPTZ
);

-- Create indexes
CREATE INDEX idx_transactions_user_id ON transactions(user_id);
CREATE INDEX idx_transactions_status ON transactions(status);
CREATE INDEX idx_transactions_chain_id ON transactions(chain_id);
CREATE INDEX idx_transactions_privacy_level ON transactions(privacy_level);
CREATE INDEX idx_transactions_created_at ON transactions(created_at DESC);
CREATE INDEX idx_transactions_chain_tx_hash ON transactions(chain_tx_hash) WHERE chain_tx_hash IS NOT NULL;
CREATE INDEX idx_transactions_proof_id ON transactions(proof_id) WHERE proof_id IS NOT NULL;

-- Composite indexes for common queries
CREATE INDEX idx_transactions_user_status ON transactions(user_id, status);
CREATE INDEX idx_transactions_user_created ON transactions(user_id, created_at DESC);

-- Create updated_at trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_transactions_updated_at
    BEFORE UPDATE ON transactions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Create proof requests table (for tracking async proof generation)
CREATE TABLE IF NOT EXISTS proof_requests (
    id UUID PRIMARY KEY,
    transaction_id UUID NOT NULL REFERENCES transactions(id),
    status VARCHAR(32) NOT NULL DEFAULT 'queued',
    priority VARCHAR(16) NOT NULL DEFAULT 'normal',
    callback_url TEXT,
    proof TEXT,
    verification_key TEXT,
    public_inputs JSONB,
    generation_time_ms BIGINT,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

CREATE INDEX idx_proof_requests_transaction_id ON proof_requests(transaction_id);
CREATE INDEX idx_proof_requests_status ON proof_requests(status);
CREATE INDEX idx_proof_requests_created_at ON proof_requests(created_at DESC);

-- Create selective disclosures table
CREATE TABLE IF NOT EXISTS selective_disclosures (
    id UUID PRIMARY KEY,
    transaction_id UUID NOT NULL REFERENCES transactions(id),
    recipient_id VARCHAR(256) NOT NULL,
    fields TEXT[] NOT NULL,
    purpose TEXT NOT NULL,
    proof TEXT NOT NULL,
    expires_at TIMESTAMPTZ,
    revoked_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_selective_disclosures_transaction_id ON selective_disclosures(transaction_id);
CREATE INDEX idx_selective_disclosures_recipient_id ON selective_disclosures(recipient_id);
CREATE INDEX idx_selective_disclosures_expires_at ON selective_disclosures(expires_at) WHERE expires_at IS NOT NULL;

-- Create privacy morph history table (for audit)
CREATE TABLE IF NOT EXISTS privacy_morph_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    transaction_id UUID NOT NULL REFERENCES transactions(id),
    previous_level SMALLINT NOT NULL,
    new_level SMALLINT NOT NULL,
    reason TEXT,
    morphed_by UUID,
    morphed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_privacy_morph_history_transaction_id ON privacy_morph_history(transaction_id);

-- Create batch operations table
CREATE TABLE IF NOT EXISTS batch_operations (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL,
    total_count INTEGER NOT NULL,
    created_count INTEGER NOT NULL DEFAULT 0,
    failed_count INTEGER NOT NULL DEFAULT 0,
    generate_proofs BOOLEAN NOT NULL DEFAULT false,
    atomic BOOLEAN NOT NULL DEFAULT false,
    status VARCHAR(32) NOT NULL DEFAULT 'processing',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

CREATE INDEX idx_batch_operations_user_id ON batch_operations(user_id);
CREATE INDEX idx_batch_operations_status ON batch_operations(status);

-- Create batch transaction mapping table
CREATE TABLE IF NOT EXISTS batch_transactions (
    batch_id UUID NOT NULL REFERENCES batch_operations(id),
    transaction_id UUID NOT NULL REFERENCES transactions(id),
    index_in_batch INTEGER NOT NULL,
    success BOOLEAN NOT NULL,
    error_message TEXT,
    PRIMARY KEY (batch_id, transaction_id)
);

CREATE INDEX idx_batch_transactions_batch_id ON batch_transactions(batch_id);
