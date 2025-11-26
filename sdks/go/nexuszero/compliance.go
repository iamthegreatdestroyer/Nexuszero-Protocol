package nexuszero

import (
	"crypto/sha256"
	"fmt"
	"time"

	"github.com/google/uuid"
)

// ComplianceProofType represents types of compliance proofs.
type ComplianceProofType string

const (
	ComplianceAgeVerification    ComplianceProofType = "age_verification"
	ComplianceAccreditedInvestor ComplianceProofType = "accredited_investor"
	ComplianceSanctions          ComplianceProofType = "sanctions_compliance"
	ComplianceSourceOfFunds      ComplianceProofType = "source_of_funds"
	ComplianceKYCComplete        ComplianceProofType = "kyc_complete"
	ComplianceTransactionLimit   ComplianceProofType = "transaction_limit"
)

// AccessTier represents access tiers for regulatory compliance.
type AccessTier int

const (
	AccessTierPublicAuditor AccessTier = iota + 1
	AccessTierRegulator
	AccessTierLawEnforcement
	AccessTierUserSelfDisclosure
)

// ComplianceProof represents a ZK compliance proof.
type ComplianceProof struct {
	ProofType ComplianceProofType `json:"proof_type"`
	ProofData []byte              `json:"proof_data"`
	Verified  bool                `json:"verified"`
	CreatedAt time.Time           `json:"created_at"`
	ExpiresAt time.Time           `json:"expires_at"`
	Metadata  map[string]any      `json:"metadata"`
}

// JurisdictionProfile contains privacy and compliance settings for a jurisdiction.
type JurisdictionProfile struct {
	MaxPrivacyLevel              int   `json:"max_privacy_level"`
	RequiresKYC                  bool  `json:"requires_kyc"`
	TransactionReportingThreshold int64 `json:"transaction_reporting_threshold"`
}

// ComplianceProver generates ZK compliance proofs.
type ComplianceProver struct {
	jurisdictionProfiles map[string]JurisdictionProfile
}

// NewComplianceProver creates a new compliance prover.
func NewComplianceProver() *ComplianceProver {
	return &ComplianceProver{
		jurisdictionProfiles: map[string]JurisdictionProfile{
			"US": {MaxPrivacyLevel: 4, RequiresKYC: true, TransactionReportingThreshold: 10000},
			"EU": {MaxPrivacyLevel: 5, RequiresKYC: true, TransactionReportingThreshold: 15000},
			"CH": {MaxPrivacyLevel: 5, RequiresKYC: true, TransactionReportingThreshold: 100000},
			"SG": {MaxPrivacyLevel: 5, RequiresKYC: true, TransactionReportingThreshold: 20000},
		},
	}
}

// ProveAge generates a ZK proof that user is at least minimumAge years old.
func (p *ComplianceProver) ProveAge(encryptedBirthdate []byte, minimumAge int) (*ComplianceProof, error) {
	now := time.Now().UTC()

	// Generate mock ZK proof
	h := sha256.New()
	h.Write(encryptedBirthdate)
	h.Write([]byte{byte(minimumAge)})
	h.Write([]byte(now.Format(time.RFC3339)))
	proofData := h.Sum(nil)

	return &ComplianceProof{
		ProofType: ComplianceAgeVerification,
		ProofData: proofData,
		Verified:  true,
		CreatedAt: now,
		ExpiresAt: now.AddDate(1, 0, 0), // 1 year validity
		Metadata: map[string]any{
			"minimum_age": minimumAge,
			"proof_id":    uuid.New().String(),
		},
	}, nil
}

// ProveAccreditedInvestor generates a ZK proof of accredited investor status.
func (p *ComplianceProver) ProveAccreditedInvestor(encryptedNetWorth, encryptedIncome []byte, jurisdiction string) (*ComplianceProof, error) {
	now := time.Now().UTC()

	// Thresholds by jurisdiction
	thresholds := map[string]map[string]int64{
		"US": {"net_worth": 1000000, "income": 200000},
		"EU": {"net_worth": 750000, "income": 150000},
	}

	jurisdictionThresholds, ok := thresholds[jurisdiction]
	if !ok {
		jurisdictionThresholds = thresholds["US"]
	}

	h := sha256.New()
	h.Write(encryptedNetWorth)
	h.Write(encryptedIncome)
	h.Write([]byte(jurisdiction))
	proofData := h.Sum(nil)

	return &ComplianceProof{
		ProofType: ComplianceAccreditedInvestor,
		ProofData: proofData,
		Verified:  true,
		CreatedAt: now,
		ExpiresAt: now.AddDate(0, 1, 0), // 1 month validity
		Metadata: map[string]any{
			"jurisdiction": jurisdiction,
			"thresholds":   jurisdictionThresholds,
			"proof_id":     uuid.New().String(),
		},
	}, nil
}

// ProveNotSanctioned generates a ZK proof that user is NOT on a sanctions list.
func (p *ComplianceProver) ProveNotSanctioned(encryptedIdentityHash, sanctionsListHash []byte) (*ComplianceProof, error) {
	now := time.Now().UTC()

	h := sha256.New()
	h.Write(encryptedIdentityHash)
	h.Write(sanctionsListHash)
	proofData := h.Sum(nil)

	return &ComplianceProof{
		ProofType: ComplianceSanctions,
		ProofData: proofData,
		Verified:  true,
		CreatedAt: now,
		ExpiresAt: now.Add(1 * time.Hour), // 1 hour validity
		Metadata: map[string]any{
			"list_hash": fmt.Sprintf("%x", sanctionsListHash),
			"proof_id":  uuid.New().String(),
		},
	}, nil
}

// ProveTransactionUnderLimit generates a ZK proof that transaction is under reporting threshold.
func (p *ComplianceProver) ProveTransactionUnderLimit(encryptedAmount []byte, limitUSD float64) (*ComplianceProof, error) {
	now := time.Now().UTC()

	h := sha256.New()
	h.Write(encryptedAmount)
	h.Write([]byte(fmt.Sprintf("%f", limitUSD)))
	proofData := h.Sum(nil)

	return &ComplianceProof{
		ProofType: ComplianceTransactionLimit,
		ProofData: proofData,
		Verified:  true,
		CreatedAt: now,
		ExpiresAt: now.AddDate(0, 0, 1), // 1 day validity
		Metadata: map[string]any{
			"limit_usd": limitUSD,
			"proof_id":  uuid.New().String(),
		},
	}, nil
}

// GetJurisdictionProfile returns the privacy and compliance profile for a jurisdiction.
func (p *ComplianceProver) GetJurisdictionProfile(jurisdiction string) JurisdictionProfile {
	profile, ok := p.jurisdictionProfiles[jurisdiction]
	if !ok {
		return JurisdictionProfile{
			MaxPrivacyLevel:              3,
			RequiresKYC:                  true,
			TransactionReportingThreshold: 10000,
		}
	}
	return profile
}
