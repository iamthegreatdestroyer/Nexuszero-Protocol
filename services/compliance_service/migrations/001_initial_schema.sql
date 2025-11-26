-- Compliance Service Initial Schema
-- NexusZero Protocol - Phase 1
-- Regulatory Compliance Layer (RCL)

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================================================
-- COMPLIANCE JURISDICTIONS
-- ============================================================================

CREATE TABLE jurisdictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    code VARCHAR(10) NOT NULL UNIQUE,
    name VARCHAR(255) NOT NULL,
    region VARCHAR(100),
    
    -- Regulatory parameters
    aml_threshold_usd DECIMAL(20, 2) NOT NULL DEFAULT 10000.00,
    kyc_required BOOLEAN NOT NULL DEFAULT true,
    travel_rule_threshold_usd DECIMAL(20, 2) NOT NULL DEFAULT 3000.00,
    privacy_coin_allowed BOOLEAN NOT NULL DEFAULT true,
    
    -- Feature flags
    real_time_monitoring BOOLEAN NOT NULL DEFAULT true,
    enhanced_due_diligence_threshold_usd DECIMAL(20, 2),
    
    -- Configuration
    config JSONB NOT NULL DEFAULT '{}',
    
    -- Status
    is_active BOOLEAN NOT NULL DEFAULT true,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_jurisdictions_code ON jurisdictions(code);
CREATE INDEX idx_jurisdictions_active ON jurisdictions(is_active) WHERE is_active = true;

-- ============================================================================
-- COMPLIANCE RULES
-- ============================================================================

CREATE TYPE rule_category AS ENUM (
    'aml',
    'kyc',
    'sanctions',
    'travel_rule',
    'reporting',
    'threshold',
    'velocity',
    'pattern',
    'custom'
);

CREATE TYPE rule_severity AS ENUM (
    'low',
    'medium',
    'high',
    'critical'
);

CREATE TABLE compliance_rules (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    jurisdiction_id UUID REFERENCES jurisdictions(id),
    
    -- Rule identification
    name VARCHAR(255) NOT NULL,
    code VARCHAR(100) NOT NULL,
    category rule_category NOT NULL,
    severity rule_severity NOT NULL DEFAULT 'medium',
    
    -- Rule definition
    description TEXT,
    expression TEXT NOT NULL,  -- Rule evaluation expression
    
    -- Thresholds
    threshold_value DECIMAL(20, 8),
    threshold_unit VARCHAR(50),
    
    -- Actions
    action_on_match VARCHAR(50) NOT NULL DEFAULT 'flag',  -- flag, block, alert, escalate
    
    -- Status
    is_active BOOLEAN NOT NULL DEFAULT true,
    version INTEGER NOT NULL DEFAULT 1,
    
    -- Timestamps
    effective_from TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    effective_until TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    UNIQUE(jurisdiction_id, code, version)
);

CREATE INDEX idx_compliance_rules_jurisdiction ON compliance_rules(jurisdiction_id);
CREATE INDEX idx_compliance_rules_category ON compliance_rules(category);
CREATE INDEX idx_compliance_rules_active ON compliance_rules(is_active) WHERE is_active = true;
CREATE INDEX idx_compliance_rules_severity ON compliance_rules(severity);

-- ============================================================================
-- COMPLIANCE CHECKS
-- ============================================================================

CREATE TYPE compliance_status AS ENUM (
    'pending',
    'approved',
    'rejected',
    'flagged',
    'escalated',
    'under_review',
    'expired'
);

CREATE TABLE compliance_checks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Subject identification
    transaction_id UUID,
    account_id UUID,
    entity_id UUID,
    
    -- Check details
    check_type VARCHAR(100) NOT NULL,
    jurisdiction_id UUID REFERENCES jurisdictions(id),
    
    -- Results
    status compliance_status NOT NULL DEFAULT 'pending',
    risk_score DECIMAL(5, 4),  -- 0.0000 to 1.0000
    risk_level VARCHAR(20),
    
    -- Matched rules
    matched_rules JSONB NOT NULL DEFAULT '[]',
    violation_count INTEGER NOT NULL DEFAULT 0,
    
    -- Details
    details JSONB NOT NULL DEFAULT '{}',
    notes TEXT,
    
    -- Review
    reviewed_by UUID,
    reviewed_at TIMESTAMP WITH TIME ZONE,
    review_decision VARCHAR(50),
    review_notes TEXT,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_compliance_checks_transaction ON compliance_checks(transaction_id);
CREATE INDEX idx_compliance_checks_account ON compliance_checks(account_id);
CREATE INDEX idx_compliance_checks_status ON compliance_checks(status);
CREATE INDEX idx_compliance_checks_created ON compliance_checks(created_at DESC);
CREATE INDEX idx_compliance_checks_risk ON compliance_checks(risk_score) WHERE risk_score IS NOT NULL;

-- ============================================================================
-- RISK ASSESSMENTS
-- ============================================================================

CREATE TYPE risk_level_enum AS ENUM (
    'minimal',
    'low',
    'moderate',
    'elevated',
    'high',
    'critical'
);

CREATE TABLE risk_assessments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Subject
    entity_type VARCHAR(50) NOT NULL,  -- transaction, account, address, entity
    entity_id UUID NOT NULL,
    
    -- Assessment details
    jurisdiction_id UUID REFERENCES jurisdictions(id),
    assessment_type VARCHAR(100) NOT NULL,
    
    -- Risk scores
    overall_risk_score DECIMAL(5, 4) NOT NULL,
    risk_level risk_level_enum NOT NULL,
    
    -- Component scores
    transaction_risk_score DECIMAL(5, 4),
    counterparty_risk_score DECIMAL(5, 4),
    behavioral_risk_score DECIMAL(5, 4),
    velocity_risk_score DECIMAL(5, 4),
    geographic_risk_score DECIMAL(5, 4),
    
    -- Risk factors
    risk_factors JSONB NOT NULL DEFAULT '[]',
    mitigating_factors JSONB NOT NULL DEFAULT '[]',
    
    -- Details
    methodology_version VARCHAR(50) NOT NULL DEFAULT 'v1.0',
    confidence_score DECIMAL(5, 4),
    
    -- Timestamps
    assessed_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    valid_until TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_risk_assessments_entity ON risk_assessments(entity_type, entity_id);
CREATE INDEX idx_risk_assessments_level ON risk_assessments(risk_level);
CREATE INDEX idx_risk_assessments_score ON risk_assessments(overall_risk_score DESC);
CREATE INDEX idx_risk_assessments_assessed ON risk_assessments(assessed_at DESC);

-- ============================================================================
-- KYC RECORDS
-- ============================================================================

CREATE TYPE kyc_status AS ENUM (
    'not_started',
    'pending',
    'documents_submitted',
    'under_review',
    'additional_info_required',
    'approved',
    'rejected',
    'expired',
    'suspended'
);

CREATE TYPE kyc_level AS ENUM (
    'none',
    'basic',
    'standard',
    'enhanced',
    'institutional'
);

CREATE TABLE kyc_records (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Subject
    account_id UUID NOT NULL,
    entity_type VARCHAR(50) NOT NULL DEFAULT 'individual',  -- individual, business, institution
    
    -- Status
    status kyc_status NOT NULL DEFAULT 'not_started',
    level kyc_level NOT NULL DEFAULT 'none',
    
    -- Identity data (encrypted reference)
    identity_hash BYTEA NOT NULL,
    identity_verified BOOLEAN NOT NULL DEFAULT false,
    
    -- Verification details
    verification_method VARCHAR(100),
    verification_provider VARCHAR(100),
    verification_reference VARCHAR(255),
    
    -- Documents
    documents_submitted JSONB NOT NULL DEFAULT '[]',
    documents_verified JSONB NOT NULL DEFAULT '[]',
    
    -- Risk assessment
    risk_score DECIMAL(5, 4),
    risk_level risk_level_enum,
    pep_status BOOLEAN,  -- Politically Exposed Person
    sanctions_match BOOLEAN,
    
    -- Review
    reviewed_by UUID,
    reviewed_at TIMESTAMP WITH TIME ZONE,
    rejection_reason TEXT,
    
    -- Validity
    verified_at TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX idx_kyc_records_account ON kyc_records(account_id);
CREATE INDEX idx_kyc_records_status ON kyc_records(status);
CREATE INDEX idx_kyc_records_level ON kyc_records(level);
CREATE INDEX idx_kyc_records_expires ON kyc_records(expires_at) WHERE expires_at IS NOT NULL;

-- ============================================================================
-- SUSPICIOUS ACTIVITY REPORTS (SARs)
-- ============================================================================

CREATE TYPE sar_status AS ENUM (
    'draft',
    'pending_review',
    'approved',
    'filed',
    'acknowledged',
    'rejected',
    'amended',
    'withdrawn'
);

CREATE TYPE sar_priority AS ENUM (
    'low',
    'normal',
    'high',
    'urgent',
    'critical'
);

CREATE TABLE suspicious_activity_reports (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Reference
    reference_number VARCHAR(100) NOT NULL UNIQUE,
    
    -- Classification
    status sar_status NOT NULL DEFAULT 'draft',
    priority sar_priority NOT NULL DEFAULT 'normal',
    
    -- Subject information
    subject_type VARCHAR(50) NOT NULL,
    subject_id UUID,
    subject_details JSONB NOT NULL DEFAULT '{}',
    
    -- Related transactions
    transaction_ids UUID[] NOT NULL DEFAULT '{}',
    total_amount_usd DECIMAL(20, 2),
    
    -- Activity details
    activity_type VARCHAR(100) NOT NULL,
    activity_date_from TIMESTAMP WITH TIME ZONE,
    activity_date_to TIMESTAMP WITH TIME ZONE,
    
    -- Narrative
    narrative TEXT NOT NULL,
    suspicious_indicators JSONB NOT NULL DEFAULT '[]',
    
    -- Jurisdiction
    jurisdiction_id UUID REFERENCES jurisdictions(id),
    filing_jurisdiction VARCHAR(10),
    
    -- Filing information
    filed_at TIMESTAMP WITH TIME ZONE,
    filed_by UUID,
    filing_reference VARCHAR(255),
    acknowledgment_reference VARCHAR(255),
    
    -- Attachments
    attachments JSONB NOT NULL DEFAULT '[]',
    
    -- Review workflow
    prepared_by UUID,
    prepared_at TIMESTAMP WITH TIME ZONE,
    reviewed_by UUID,
    reviewed_at TIMESTAMP WITH TIME ZONE,
    approved_by UUID,
    approved_at TIMESTAMP WITH TIME ZONE,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    -- Search
    search_vector TSVECTOR
);

CREATE INDEX idx_sar_status ON suspicious_activity_reports(status);
CREATE INDEX idx_sar_priority ON suspicious_activity_reports(priority);
CREATE INDEX idx_sar_subject ON suspicious_activity_reports(subject_type, subject_id);
CREATE INDEX idx_sar_filed ON suspicious_activity_reports(filed_at DESC) WHERE filed_at IS NOT NULL;
CREATE INDEX idx_sar_created ON suspicious_activity_reports(created_at DESC);
CREATE INDEX idx_sar_search ON suspicious_activity_reports USING GIN(search_vector);

-- ============================================================================
-- TRAVEL RULE MESSAGES
-- ============================================================================

CREATE TYPE travel_rule_direction AS ENUM (
    'outbound',
    'inbound'
);

CREATE TYPE travel_rule_status AS ENUM (
    'pending',
    'sent',
    'received',
    'acknowledged',
    'accepted',
    'rejected',
    'expired'
);

CREATE TABLE travel_rule_messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Transaction reference
    transaction_id UUID NOT NULL,
    
    -- Direction
    direction travel_rule_direction NOT NULL,
    status travel_rule_status NOT NULL DEFAULT 'pending',
    
    -- Originator information (encrypted)
    originator_vasp_id VARCHAR(255),
    originator_name_hash BYTEA,
    originator_account_hash BYTEA,
    originator_address_hash BYTEA,
    
    -- Beneficiary information (encrypted)
    beneficiary_vasp_id VARCHAR(255),
    beneficiary_name_hash BYTEA,
    beneficiary_account_hash BYTEA,
    beneficiary_address_hash BYTEA,
    
    -- Transaction details
    amount DECIMAL(30, 18) NOT NULL,
    asset VARCHAR(50) NOT NULL,
    amount_usd DECIMAL(20, 2),
    
    -- Protocol details
    protocol_version VARCHAR(20) NOT NULL DEFAULT 'v1',
    message_hash BYTEA NOT NULL,
    encrypted_payload BYTEA NOT NULL,
    
    -- Counterparty VASP
    counterparty_vasp_id VARCHAR(255) NOT NULL,
    counterparty_vasp_name VARCHAR(255),
    counterparty_verified BOOLEAN NOT NULL DEFAULT false,
    
    -- Response
    response_message_id UUID,
    response_received_at TIMESTAMP WITH TIME ZONE,
    response_status VARCHAR(50),
    
    -- Timestamps
    sent_at TIMESTAMP WITH TIME ZONE,
    received_at TIMESTAMP WITH TIME ZONE,
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_travel_rule_transaction ON travel_rule_messages(transaction_id);
CREATE INDEX idx_travel_rule_direction ON travel_rule_messages(direction);
CREATE INDEX idx_travel_rule_status ON travel_rule_messages(status);
CREATE INDEX idx_travel_rule_counterparty ON travel_rule_messages(counterparty_vasp_id);
CREATE INDEX idx_travel_rule_created ON travel_rule_messages(created_at DESC);

-- ============================================================================
-- AUDIT LOG
-- ============================================================================

CREATE TABLE compliance_audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Actor
    actor_id UUID,
    actor_type VARCHAR(50) NOT NULL,  -- user, system, service
    
    -- Action
    action VARCHAR(100) NOT NULL,
    action_category VARCHAR(50) NOT NULL,
    
    -- Target
    target_type VARCHAR(50) NOT NULL,
    target_id UUID,
    
    -- Details
    details JSONB NOT NULL DEFAULT '{}',
    changes JSONB,  -- before/after for updates
    
    -- Context
    ip_address INET,
    user_agent TEXT,
    session_id VARCHAR(255),
    correlation_id UUID,
    
    -- Result
    success BOOLEAN NOT NULL,
    error_message TEXT,
    
    -- Timestamp
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_audit_log_actor ON compliance_audit_log(actor_id, actor_type);
CREATE INDEX idx_audit_log_action ON compliance_audit_log(action, action_category);
CREATE INDEX idx_audit_log_target ON compliance_audit_log(target_type, target_id);
CREATE INDEX idx_audit_log_created ON compliance_audit_log(created_at DESC);
CREATE INDEX idx_audit_log_correlation ON compliance_audit_log(correlation_id) WHERE correlation_id IS NOT NULL;

-- Partitioning for audit log (if volume warrants)
-- Consider partitioning by created_at for high-volume deployments

-- ============================================================================
-- COMPLIANCE REPORTS
-- ============================================================================

CREATE TYPE report_type AS ENUM (
    'daily_summary',
    'weekly_summary',
    'monthly_summary',
    'risk_report',
    'sar_summary',
    'kyc_status',
    'travel_rule_summary',
    'jurisdiction_report',
    'custom'
);

CREATE TYPE report_format AS ENUM (
    'json',
    'pdf',
    'csv',
    'xlsx'
);

CREATE TABLE compliance_reports (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Report identification
    report_type report_type NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    
    -- Parameters
    parameters JSONB NOT NULL DEFAULT '{}',
    period_start TIMESTAMP WITH TIME ZONE,
    period_end TIMESTAMP WITH TIME ZONE,
    jurisdiction_id UUID REFERENCES jurisdictions(id),
    
    -- Generation
    format report_format NOT NULL DEFAULT 'json',
    generated_by UUID,
    generated_at TIMESTAMP WITH TIME ZONE,
    generation_duration_ms INTEGER,
    
    -- Storage
    file_path TEXT,
    file_size_bytes BIGINT,
    checksum VARCHAR(128),
    
    -- Content (for smaller reports)
    content JSONB,
    
    -- Status
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    error_message TEXT,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_compliance_reports_type ON compliance_reports(report_type);
CREATE INDEX idx_compliance_reports_generated ON compliance_reports(generated_at DESC);
CREATE INDEX idx_compliance_reports_jurisdiction ON compliance_reports(jurisdiction_id);

-- ============================================================================
-- SANCTIONS LISTS
-- ============================================================================

CREATE TABLE sanctions_entries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Source
    list_source VARCHAR(100) NOT NULL,  -- OFAC, EU, UN, etc.
    list_program VARCHAR(100),
    
    -- Entry details
    entry_type VARCHAR(50) NOT NULL,  -- individual, entity, vessel, aircraft
    primary_name VARCHAR(500) NOT NULL,
    aliases JSONB NOT NULL DEFAULT '[]',
    
    -- Identifiers
    addresses JSONB NOT NULL DEFAULT '[]',
    id_numbers JSONB NOT NULL DEFAULT '[]',
    
    -- Crypto-specific
    wallet_addresses JSONB NOT NULL DEFAULT '[]',
    
    -- Dates
    listed_date DATE,
    delisted_date DATE,
    
    -- Sanctions details
    programs JSONB NOT NULL DEFAULT '[]',
    remarks TEXT,
    
    -- Status
    is_active BOOLEAN NOT NULL DEFAULT true,
    
    -- Search
    search_vector TSVECTOR,
    
    -- Timestamps
    source_updated_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_sanctions_source ON sanctions_entries(list_source);
CREATE INDEX idx_sanctions_type ON sanctions_entries(entry_type);
CREATE INDEX idx_sanctions_active ON sanctions_entries(is_active) WHERE is_active = true;
CREATE INDEX idx_sanctions_search ON sanctions_entries USING GIN(search_vector);
CREATE INDEX idx_sanctions_wallets ON sanctions_entries USING GIN(wallet_addresses);

-- ============================================================================
-- FUNCTIONS AND TRIGGERS
-- ============================================================================

-- Update timestamp trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply update triggers
CREATE TRIGGER update_jurisdictions_updated_at
    BEFORE UPDATE ON jurisdictions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_compliance_rules_updated_at
    BEFORE UPDATE ON compliance_rules
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_compliance_checks_updated_at
    BEFORE UPDATE ON compliance_checks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_kyc_records_updated_at
    BEFORE UPDATE ON kyc_records
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_sar_updated_at
    BEFORE UPDATE ON suspicious_activity_reports
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_travel_rule_updated_at
    BEFORE UPDATE ON travel_rule_messages
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_sanctions_updated_at
    BEFORE UPDATE ON sanctions_entries
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- SAR search vector update
CREATE OR REPLACE FUNCTION sar_search_vector_update()
RETURNS TRIGGER AS $$
BEGIN
    NEW.search_vector := 
        setweight(to_tsvector('english', COALESCE(NEW.reference_number, '')), 'A') ||
        setweight(to_tsvector('english', COALESCE(NEW.narrative, '')), 'B') ||
        setweight(to_tsvector('english', COALESCE(NEW.activity_type, '')), 'C');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER sar_search_vector_trigger
    BEFORE INSERT OR UPDATE ON suspicious_activity_reports
    FOR EACH ROW EXECUTE FUNCTION sar_search_vector_update();

-- Sanctions search vector update
CREATE OR REPLACE FUNCTION sanctions_search_vector_update()
RETURNS TRIGGER AS $$
BEGIN
    NEW.search_vector := 
        setweight(to_tsvector('english', COALESCE(NEW.primary_name, '')), 'A') ||
        setweight(to_tsvector('english', COALESCE(NEW.aliases::text, '')), 'B') ||
        setweight(to_tsvector('english', COALESCE(NEW.remarks, '')), 'C');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER sanctions_search_vector_trigger
    BEFORE INSERT OR UPDATE ON sanctions_entries
    FOR EACH ROW EXECUTE FUNCTION sanctions_search_vector_update();

-- ============================================================================
-- SEED DATA
-- ============================================================================

-- Insert default jurisdictions
INSERT INTO jurisdictions (code, name, region, aml_threshold_usd, kyc_required, travel_rule_threshold_usd, privacy_coin_allowed) VALUES
    ('US', 'United States', 'North America', 10000.00, true, 3000.00, true),
    ('EU', 'European Union', 'Europe', 10000.00, true, 1000.00, true),
    ('UK', 'United Kingdom', 'Europe', 10000.00, true, 1000.00, true),
    ('SG', 'Singapore', 'Asia-Pacific', 20000.00, true, 1500.00, true),
    ('CH', 'Switzerland', 'Europe', 15000.00, true, 1000.00, true),
    ('JP', 'Japan', 'Asia-Pacific', 10000.00, true, 3000.00, false),
    ('AE', 'United Arab Emirates', 'Middle East', 15000.00, true, 1000.00, true),
    ('HK', 'Hong Kong', 'Asia-Pacific', 10000.00, true, 1000.00, true);

-- Insert default compliance rules
INSERT INTO compliance_rules (jurisdiction_id, name, code, category, severity, expression, threshold_value, threshold_unit, action_on_match) VALUES
    ((SELECT id FROM jurisdictions WHERE code = 'US'), 'Large Transaction Reporting', 'US-CTR-10K', 'threshold', 'medium', 'amount_usd >= threshold', 10000.00, 'USD', 'flag'),
    ((SELECT id FROM jurisdictions WHERE code = 'US'), 'Travel Rule Threshold', 'US-TR-3K', 'travel_rule', 'medium', 'amount_usd >= threshold', 3000.00, 'USD', 'alert'),
    ((SELECT id FROM jurisdictions WHERE code = 'US'), 'OFAC Sanctions Check', 'US-OFAC', 'sanctions', 'critical', 'sanctions_match == true', NULL, NULL, 'block'),
    ((SELECT id FROM jurisdictions WHERE code = 'EU'), 'AMLD Threshold', 'EU-AMLD-10K', 'threshold', 'medium', 'amount_usd >= threshold', 10000.00, 'EUR', 'flag'),
    ((SELECT id FROM jurisdictions WHERE code = 'EU'), 'Travel Rule (TRAV)', 'EU-TRAV-1K', 'travel_rule', 'medium', 'amount_usd >= threshold', 1000.00, 'EUR', 'alert'),
    (NULL, 'High Velocity Alert', 'GLOBAL-VEL-24H', 'velocity', 'high', 'tx_count_24h > threshold', 100.00, 'count', 'escalate'),
    (NULL, 'New Account Large Transfer', 'GLOBAL-NEW-LT', 'pattern', 'high', 'account_age_days < 7 AND amount_usd > 5000', NULL, NULL, 'flag');

-- ============================================================================
-- GRANTS
-- ============================================================================

-- Grant appropriate permissions (adjust based on your user setup)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO compliance_service;
-- GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO compliance_service;

COMMENT ON TABLE jurisdictions IS 'Regulatory jurisdictions with their specific compliance parameters';
COMMENT ON TABLE compliance_rules IS 'Configurable rules for compliance checking per jurisdiction';
COMMENT ON TABLE compliance_checks IS 'Results of compliance checks on transactions and entities';
COMMENT ON TABLE risk_assessments IS 'Detailed risk assessments with component scores';
COMMENT ON TABLE kyc_records IS 'KYC status and verification records for accounts';
COMMENT ON TABLE suspicious_activity_reports IS 'SAR filings and workflow tracking';
COMMENT ON TABLE travel_rule_messages IS 'Travel rule message exchange with counterparty VASPs';
COMMENT ON TABLE compliance_audit_log IS 'Comprehensive audit trail for all compliance actions';
COMMENT ON TABLE compliance_reports IS 'Generated compliance reports and summaries';
COMMENT ON TABLE sanctions_entries IS 'Cached sanctions list entries for screening';
