-- ============================================================
-- NexusZero Protocol Database Migration
-- Version: 001
-- Description: Initial schema creation
-- ============================================================

-- Migration metadata
CREATE TABLE IF NOT EXISTS schema_migrations (
    version VARCHAR(255) PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    description TEXT
);

-- Check if migration already applied
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM schema_migrations WHERE version = '001') THEN
        RAISE EXCEPTION 'Migration 001 already applied';
    END IF;
END $$;

-- Run the initial schema
\i ../schemas/001_initial.sql

-- Record migration
INSERT INTO schema_migrations (version, description)
VALUES ('001', 'Initial schema with users, transactions, proofs, compliance, bridge, and prover nodes tables');

-- Output success
DO $$
BEGIN
    RAISE NOTICE 'Migration 001 applied successfully';
END $$;
