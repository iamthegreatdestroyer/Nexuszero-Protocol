package nexuszero

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"time"

	"github.com/google/uuid"
)

// PrivacyParameters holds configuration for a specific privacy level.
type PrivacyParameters struct {
	LatticeN         int     `json:"lattice_n"`
	ModulusQ         uint64  `json:"modulus_q"`
	Sigma            float64 `json:"sigma"`
	SecurityBits     int     `json:"security_bits"`
	ProofStrategy    string  `json:"proof_strategy"`
	AnonymitySetSize *int    `json:"anonymity_set_size,omitempty"`
	DecoyCount       *int    `json:"decoy_count,omitempty"`
}

// TransactionContext holds context for privacy level recommendations.
type TransactionContext struct {
	ValueUSD           float64 `json:"value_usd"`
	RequiresCompliance bool    `json:"requires_compliance"`
	PreferredLevel     *int    `json:"preferred_level,omitempty"`
	RiskScore          float64 `json:"risk_score"`
	Jurisdiction       string  `json:"jurisdiction"`
	CounterpartyKnown  bool    `json:"counterparty_known"`
}

// PrivacyRecommendation is the result of privacy level recommendation.
type PrivacyRecommendation struct {
	Level               PrivacyLevel      `json:"level"`
	Parameters          PrivacyParameters `json:"parameters"`
	Reasons             []string          `json:"reasons"`
	EstimatedProofTimeMs int              `json:"estimated_proof_time_ms"`
	EstimatedCostGas    uint64            `json:"estimated_cost_gas"`
}

// PrivacyEngine provides adaptive privacy morphing functionality.
type PrivacyEngine struct {
	parameters map[PrivacyLevel]PrivacyParameters
	gasCosts   map[PrivacyLevel]uint64
}

// NewPrivacyEngine creates a new privacy engine with default parameters.
func NewPrivacyEngine() *PrivacyEngine {
	decoy3 := 3
	decoy7 := 7
	decoy15 := 15
	decoy31 := 31
	decoy63 := 63
	anon16 := 16
	anon64 := 64
	anon256 := 256

	return &PrivacyEngine{
		parameters: map[PrivacyLevel]PrivacyParameters{
			PrivacyTransparent: {
				LatticeN:      0,
				ModulusQ:      0,
				Sigma:         0.0,
				SecurityBits:  0,
				ProofStrategy: "none",
			},
			PrivacyPseudonymous: {
				LatticeN:      256,
				ModulusQ:      12289,
				Sigma:         3.2,
				SecurityBits:  80,
				ProofStrategy: "bulletproofs",
				DecoyCount:    &decoy3,
			},
			PrivacyConfidential: {
				LatticeN:      512,
				ModulusQ:      12289,
				Sigma:         3.2,
				SecurityBits:  128,
				ProofStrategy: "bulletproofs",
				DecoyCount:    &decoy7,
			},
			PrivacyPrivate: {
				LatticeN:         1024,
				ModulusQ:         40961,
				Sigma:            3.2,
				SecurityBits:     192,
				ProofStrategy:    "quantum_lattice_pkc",
				AnonymitySetSize: &anon16,
				DecoyCount:       &decoy15,
			},
			PrivacyAnonymous: {
				LatticeN:         2048,
				ModulusQ:         65537,
				Sigma:            3.2,
				SecurityBits:     256,
				ProofStrategy:    "quantum_lattice_pkc",
				AnonymitySetSize: &anon64,
				DecoyCount:       &decoy31,
			},
			PrivacySovereign: {
				LatticeN:         4096,
				ModulusQ:         786433,
				Sigma:            3.2,
				SecurityBits:     256,
				ProofStrategy:    "hybrid_zksnark_lattice",
				AnonymitySetSize: &anon256,
				DecoyCount:       &decoy63,
			},
		},
		gasCosts: map[PrivacyLevel]uint64{
			PrivacyTransparent:  21000,
			PrivacyPseudonymous: 50000,
			PrivacyConfidential: 100000,
			PrivacyPrivate:      200000,
			PrivacyAnonymous:    350000,
			PrivacySovereign:    500000,
		},
	}
}

// GetParameters returns the parameters for a specific privacy level.
func (e *PrivacyEngine) GetParameters(level PrivacyLevel) (PrivacyParameters, error) {
	params, ok := e.parameters[level]
	if !ok {
		return PrivacyParameters{}, fmt.Errorf("invalid privacy level: %d", level)
	}
	return params, nil
}

// Recommend returns a privacy level recommendation based on context.
func (e *PrivacyEngine) Recommend(ctx TransactionContext) PrivacyRecommendation {
	recommendedLevel := PrivacyPrivate // Default
	var reasons []string

	// Regulatory considerations
	if ctx.RequiresCompliance {
		if recommendedLevel > PrivacyPrivate {
			recommendedLevel = PrivacyPrivate
		}
		reasons = append(reasons, "Regulatory compliance limits maximum privacy to Level 3")
	}

	// Transaction value considerations
	if ctx.ValueUSD > 10000.0 {
		if recommendedLevel < PrivacyAnonymous {
			recommendedLevel = PrivacyAnonymous
		}
		reasons = append(reasons, "High-value transaction ($10k+) benefits from Anonymous privacy")
	}
	if ctx.ValueUSD > 100000.0 {
		recommendedLevel = PrivacySovereign
		reasons = append(reasons, "Very high-value transaction ($100k+) warrants Sovereign privacy")
	}

	// User preference
	if ctx.PreferredLevel != nil {
		recommendedLevel = PrivacyLevel(*ctx.PreferredLevel)
		reasons = append(reasons, fmt.Sprintf("User preference: Level %d", *ctx.PreferredLevel))
	}

	// Risk score adjustment
	if ctx.RiskScore > 0.7 {
		if recommendedLevel > PrivacyConfidential {
			recommendedLevel = PrivacyConfidential
		}
		reasons = append(reasons, "Elevated risk score (>0.7) reduces maximum privacy")
	}

	params, _ := e.GetParameters(recommendedLevel)

	return PrivacyRecommendation{
		Level:               recommendedLevel,
		Parameters:          params,
		Reasons:             reasons,
		EstimatedProofTimeMs: recommendedLevel.EstimatedProofTimeMs(),
		EstimatedCostGas:    e.gasCosts[recommendedLevel],
	}
}

// CanMorph checks if morphing between privacy levels is possible.
func (e *PrivacyEngine) CanMorph(fromLevel, toLevel PrivacyLevel) (bool, string) {
	if fromLevel > PrivacySovereign || toLevel > PrivacySovereign {
		return false, "Invalid privacy level"
	}

	if toLevel >= fromLevel {
		return true, "Increasing or maintaining privacy level is always allowed"
	}

	return true, "Decreasing privacy must be done incrementally"
}

// GetMorphPath returns the path for morphing between levels.
func (e *PrivacyEngine) GetMorphPath(fromLevel, toLevel PrivacyLevel) []PrivacyLevel {
	if toLevel > fromLevel {
		// Increasing privacy - one step
		return []PrivacyLevel{toLevel}
	}

	// Decreasing privacy - incremental
	var path []PrivacyLevel
	for level := fromLevel - 1; level >= toLevel; level-- {
		path = append(path, level)
	}
	return path
}

// LocalProofResult is the result of local proof generation.
type LocalProofResult struct {
	ProofID          string    `json:"proof_id"`
	ProofData        []byte    `json:"proof_data"`
	PrivacyLevel     int       `json:"privacy_level"`
	GenerationTimeMs int64     `json:"generation_time_ms"`
	QualityScore     float64   `json:"quality_score"`
	GeneratedAt      time.Time `json:"generated_at"`
}

// LocalProofGenerator generates proofs locally (for testing).
type LocalProofGenerator struct {
	engine *PrivacyEngine
}

// NewLocalProofGenerator creates a new local proof generator.
func NewLocalProofGenerator() *LocalProofGenerator {
	return &LocalProofGenerator{
		engine: NewPrivacyEngine(),
	}
}

// Generate generates a proof locally.
func (g *LocalProofGenerator) Generate(data []byte, level PrivacyLevel) (*LocalProofResult, error) {
	start := time.Now()

	// Simulate proof generation
	proofSize := 32 * (int(level) + 1)
	h := sha256.New()
	h.Write(data)
	h.Write([]byte{byte(level)})
	proofHash := h.Sum(nil)

	// Extend proof to expected size
	proofData := make([]byte, 0, proofSize)
	for len(proofData) < proofSize {
		proofData = append(proofData, proofHash...)
	}
	proofData = proofData[:proofSize]

	elapsed := time.Since(start).Milliseconds()

	return &LocalProofResult{
		ProofID:          uuid.New().String(),
		ProofData:        proofData,
		PrivacyLevel:     int(level),
		GenerationTimeMs: max(elapsed, int64(level.EstimatedProofTimeMs())),
		QualityScore:     1.0,
		GeneratedAt:      time.Now(),
	}, nil
}

// LocalProofVerifier verifies proofs locally.
type LocalProofVerifier struct{}

// NewLocalProofVerifier creates a new local proof verifier.
func NewLocalProofVerifier() *LocalProofVerifier {
	return &LocalProofVerifier{}
}

// Verify verifies a proof locally.
func (v *LocalProofVerifier) Verify(proof []byte, level *PrivacyLevel) bool {
	if len(proof) < 32 {
		return false
	}

	if level != nil {
		expectedMinSize := 32 * (int(*level) + 1)
		if len(proof) < expectedMinSize {
			return false
		}
	}

	return true
}

// ProofDataHex returns the proof data as a hex string.
func (r *LocalProofResult) ProofDataHex() string {
	return hex.EncodeToString(r.ProofData)
}

func max(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}
