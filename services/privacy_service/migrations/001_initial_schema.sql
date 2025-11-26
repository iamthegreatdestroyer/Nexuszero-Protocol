-- Privacy Service Initial Schema
-- Migration: 001_initial_schema
-- Created: 2025-01-XX

-- Privacy levels enum
CREATE TYPE privacy_level AS ENUM (
    'transparent',
    'minimal',
    'standard',
    'enhanced',
    'maximum',
    'quantum'
);

-- Proof status enum
CREATE TYPE proof_status AS ENUM (
    'pending',
    'generating',
    'verified',
    'failed',
    'expired'
);

-- Morph status enum
CREATE TYPE morph_status AS ENUM (
    'pending',
    'processing',
    'completed',
    'failed'
);

-- Proof type enum
CREATE TYPE proof_type AS ENUM (
    'groth16',
    'plonk',
    'bulletproofs',
    'stark',
    'custom'
);

-- Privacy profiles table
CREATE TABLE IF NOT EXISTS privacy_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    default_level privacy_level NOT NULL DEFAULT 'standard',
    preferences JSONB NOT NULL DEFAULT '{}',
    jurisdiction VARCHAR(50),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Privacy profiles indexes
CREATE INDEX idx_privacy_profiles_user_id ON privacy_profiles(user_id);
CREATE INDEX idx_privacy_profiles_jurisdiction ON privacy_profiles(jurisdiction);

-- Morphing jobs table
CREATE TABLE IF NOT EXISTS morph_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_level privacy_level NOT NULL,
    target_level privacy_level NOT NULL,
    status morph_status NOT NULL DEFAULT 'pending',
    input_hash BYTEA NOT NULL,
    output_hash BYTEA,
    proof_data BYTEA,
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    error_message TEXT
);

-- Morph jobs indexes
CREATE INDEX idx_morph_jobs_status ON morph_jobs(status);
CREATE INDEX idx_morph_jobs_created_at ON morph_jobs(created_at DESC);
CREATE INDEX idx_morph_jobs_target_level ON morph_jobs(target_level);

-- Proof generation jobs table
CREATE TABLE IF NOT EXISTS proof_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    proof_type proof_type NOT NULL,
    status proof_status NOT NULL DEFAULT 'pending',
    circuit_id VARCHAR(255),
    public_inputs BYTEA NOT NULL,
    witness_hash BYTEA,
    proof_data BYTEA,
    verification_key_hash BYTEA,
    privacy_level privacy_level NOT NULL DEFAULT 'standard',
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,
    error_message TEXT
);

-- Proof jobs indexes
CREATE INDEX idx_proof_jobs_status ON proof_jobs(status);
CREATE INDEX idx_proof_jobs_proof_type ON proof_jobs(proof_type);
CREATE INDEX idx_proof_jobs_created_at ON proof_jobs(created_at DESC);
CREATE INDEX idx_proof_jobs_circuit_id ON proof_jobs(circuit_id);
CREATE INDEX idx_proof_jobs_expires_at ON proof_jobs(expires_at) WHERE expires_at IS NOT NULL;

-- Verification keys table
CREATE TABLE IF NOT EXISTS verification_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    key_hash BYTEA NOT NULL UNIQUE,
    key_data BYTEA NOT NULL,
    proof_type proof_type NOT NULL,
    circuit_id VARCHAR(255),
    description TEXT,
    is_active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ
);

-- Verification keys indexes
CREATE INDEX idx_verification_keys_hash ON verification_keys(key_hash);
CREATE INDEX idx_verification_keys_proof_type ON verification_keys(proof_type);
CREATE INDEX idx_verification_keys_active ON verification_keys(is_active) WHERE is_active = true;

-- Selective disclosure requests table
CREATE TABLE IF NOT EXISTS disclosure_requests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    requester_id UUID NOT NULL,
    subject_id UUID NOT NULL,
    disclosure_type VARCHAR(100) NOT NULL,
    attributes JSONB NOT NULL DEFAULT '[]',
    proof_id UUID REFERENCES proof_jobs(id),
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved_at TIMESTAMPTZ,
    metadata JSONB NOT NULL DEFAULT '{}'
);

-- Disclosure requests indexes
CREATE INDEX idx_disclosure_requests_requester ON disclosure_requests(requester_id);
CREATE INDEX idx_disclosure_requests_subject ON disclosure_requests(subject_id);
CREATE INDEX idx_disclosure_requests_status ON disclosure_requests(status);
CREATE INDEX idx_disclosure_requests_expires ON disclosure_requests(expires_at) WHERE expires_at IS NOT NULL;

-- Privacy analytics table
CREATE TABLE IF NOT EXISTS privacy_analytics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    metric_type VARCHAR(100) NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    privacy_level privacy_level,
    jurisdiction VARCHAR(50),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB NOT NULL DEFAULT '{}'
);

-- Analytics indexes
CREATE INDEX idx_privacy_analytics_type ON privacy_analytics(metric_type);
CREATE INDEX idx_privacy_analytics_level ON privacy_analytics(privacy_level);
CREATE INDEX idx_privacy_analytics_timestamp ON privacy_analytics(timestamp DESC);
CREATE INDEX idx_privacy_analytics_jurisdiction ON privacy_analytics(jurisdiction);

-- Partition analytics by month for performance
-- This would typically be done with table inheritance or declarative partitioning

-- Nullifier registry for double-spend prevention
CREATE TABLE IF NOT EXISTS nullifier_registry (
    nullifier BYTEA PRIMARY KEY,
    proof_id UUID REFERENCES proof_jobs(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB NOT NULL DEFAULT '{}'
);

-- Nullifier index for fast lookup
CREATE INDEX idx_nullifier_created_at ON nullifier_registry(created_at DESC);

-- Commitment registry
CREATE TABLE IF NOT EXISTS commitment_registry (
    commitment BYTEA PRIMARY KEY,
    proof_id UUID REFERENCES proof_jobs(id),
    privacy_level privacy_level NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    spent_at TIMESTAMPTZ,
    metadata JSONB NOT NULL DEFAULT '{}'
);

-- Commitment indexes
CREATE INDEX idx_commitment_spent ON commitment_registry(spent_at) WHERE spent_at IS NOT NULL;
CREATE INDEX idx_commitment_level ON commitment_registry(privacy_level);

-- Circuit registry for ZK circuits
CREATE TABLE IF NOT EXISTS circuit_registry (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    circuit_id VARCHAR(255) NOT NULL UNIQUE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    proof_type proof_type NOT NULL,
    constraint_count INTEGER NOT NULL DEFAULT 0,
    parameters JSONB NOT NULL DEFAULT '{}',
    verification_key_id UUID REFERENCES verification_keys(id),
    is_active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Circuit registry indexes
CREATE INDEX idx_circuit_registry_circuit_id ON circuit_registry(circuit_id);
CREATE INDEX idx_circuit_registry_proof_type ON circuit_registry(proof_type);
CREATE INDEX idx_circuit_registry_active ON circuit_registry(is_active) WHERE is_active = true;

-- Audit log for privacy operations
CREATE TABLE IF NOT EXISTS privacy_audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    operation VARCHAR(100) NOT NULL,
    entity_type VARCHAR(100) NOT NULL,
    entity_id UUID NOT NULL,
    actor_id UUID,
    old_state JSONB,
    new_state JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Audit log indexes
CREATE INDEX idx_audit_log_operation ON privacy_audit_log(operation);
CREATE INDEX idx_audit_log_entity ON privacy_audit_log(entity_type, entity_id);
CREATE INDEX idx_audit_log_actor ON privacy_audit_log(actor_id);
CREATE INDEX idx_audit_log_created ON privacy_audit_log(created_at DESC);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply updated_at trigger to relevant tables
CREATE TRIGGER update_privacy_profiles_updated_at
    BEFORE UPDATE ON privacy_profiles
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_circuit_registry_updated_at
    BEFORE UPDATE ON circuit_registry
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Function to record audit log entries
CREATE OR REPLACE FUNCTION record_privacy_audit()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO privacy_audit_log (operation, entity_type, entity_id, old_state, new_state)
    VALUES (
        TG_OP,
        TG_TABLE_NAME,
        COALESCE(NEW.id, OLD.id),
        CASE WHEN TG_OP = 'DELETE' THEN to_jsonb(OLD) ELSE NULL END,
        CASE WHEN TG_OP IN ('INSERT', 'UPDATE') THEN to_jsonb(NEW) ELSE NULL END
    );
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

-- Apply audit triggers to sensitive tables
CREATE TRIGGER audit_morph_jobs
    AFTER INSERT OR UPDATE OR DELETE ON morph_jobs
    FOR EACH ROW
    EXECUTE FUNCTION record_privacy_audit();

CREATE TRIGGER audit_proof_jobs
    AFTER INSERT OR UPDATE OR DELETE ON proof_jobs
    FOR EACH ROW
    EXECUTE FUNCTION record_privacy_audit();

CREATE TRIGGER audit_disclosure_requests
    AFTER INSERT OR UPDATE OR DELETE ON disclosure_requests
    FOR EACH ROW
    EXECUTE FUNCTION record_privacy_audit();

-- Materialized view for privacy level distribution
CREATE MATERIALIZED VIEW IF NOT EXISTS privacy_level_distribution AS
SELECT
    privacy_level,
    COUNT(*) as transaction_count,
    DATE_TRUNC('day', created_at) as day
FROM proof_jobs
WHERE created_at > NOW() - INTERVAL '30 days'
GROUP BY privacy_level, DATE_TRUNC('day', created_at);

-- Index on materialized view
CREATE UNIQUE INDEX idx_privacy_level_dist ON privacy_level_distribution(privacy_level, day);

-- Refresh function for materialized view
CREATE OR REPLACE FUNCTION refresh_privacy_level_distribution()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY privacy_level_distribution;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions (adjust role names as needed)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO privacy_service;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO privacy_service;

COMMENT ON TABLE privacy_profiles IS 'User privacy preferences and default settings';
COMMENT ON TABLE morph_jobs IS 'Tracking for APM privacy morphing operations';
COMMENT ON TABLE proof_jobs IS 'ZK proof generation and verification jobs';
COMMENT ON TABLE verification_keys IS 'Registry of ZK circuit verification keys';
COMMENT ON TABLE disclosure_requests IS 'Selective disclosure requests and responses';
COMMENT ON TABLE nullifier_registry IS 'Prevents double-spending of privacy proofs';
COMMENT ON TABLE commitment_registry IS 'Registry of Pedersen-style commitments';
COMMENT ON TABLE circuit_registry IS 'ZK circuit definitions and metadata';
COMMENT ON TABLE privacy_audit_log IS 'Audit trail for all privacy operations';
