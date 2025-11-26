package nexuszero

import (
	"testing"
)

func TestPrivacyLevel(t *testing.T) {
	t.Run("String returns correct names", func(t *testing.T) {
		tests := []struct {
			level    PrivacyLevel
			expected string
		}{
			{PrivacyTransparent, "Transparent"},
			{PrivacyPseudonymous, "Pseudonymous"},
			{PrivacyConfidential, "Confidential"},
			{PrivacyPrivate, "Private"},
			{PrivacyAnonymous, "Anonymous"},
			{PrivacySovereign, "Sovereign"},
		}

		for _, tt := range tests {
			if got := tt.level.String(); got != tt.expected {
				t.Errorf("PrivacyLevel(%d).String() = %s, want %s", tt.level, got, tt.expected)
			}
		}
	})

	t.Run("SecurityBits increases with level", func(t *testing.T) {
		var prev int
		for level := PrivacyTransparent; level <= PrivacySovereign; level++ {
			bits := level.SecurityBits()
			if bits < prev {
				t.Errorf("SecurityBits for level %d (%d) < previous (%d)", level, bits, prev)
			}
			prev = bits
		}
	})

	t.Run("EstimatedProofTimeMs increases with level", func(t *testing.T) {
		var prev int
		for level := PrivacyTransparent; level <= PrivacySovereign; level++ {
			time := level.EstimatedProofTimeMs()
			if time < prev {
				t.Errorf("EstimatedProofTimeMs for level %d (%d) < previous (%d)", level, time, prev)
			}
			prev = time
		}
	})
}

func TestPrivacyEngine(t *testing.T) {
	engine := NewPrivacyEngine()

	t.Run("GetParameters returns valid params for all levels", func(t *testing.T) {
		for level := PrivacyTransparent; level <= PrivacySovereign; level++ {
			params, err := engine.GetParameters(level)
			if err != nil {
				t.Errorf("GetParameters(%d) returned error: %v", level, err)
			}
			if level > PrivacyTransparent && params.LatticeN == 0 {
				t.Errorf("GetParameters(%d) returned zero LatticeN for non-transparent level", level)
			}
		}
	})

	t.Run("Recommend returns Private by default", func(t *testing.T) {
		ctx := TransactionContext{ValueUSD: 100.0}
		rec := engine.Recommend(ctx)
		if rec.Level != PrivacyPrivate {
			t.Errorf("Default recommendation = %v, want Private", rec.Level)
		}
	})

	t.Run("Recommend increases level for high-value transactions", func(t *testing.T) {
		ctx := TransactionContext{ValueUSD: 50000.0}
		rec := engine.Recommend(ctx)
		if rec.Level < PrivacyAnonymous {
			t.Errorf("High-value recommendation = %v, want >= Anonymous", rec.Level)
		}
	})

	t.Run("Recommend limits level for compliance", func(t *testing.T) {
		ctx := TransactionContext{
			ValueUSD:           100000.0,
			RequiresCompliance: true,
		}
		rec := engine.Recommend(ctx)
		if rec.Level > PrivacyPrivate {
			t.Errorf("Compliance recommendation = %v, want <= Private", rec.Level)
		}
	})

	t.Run("Recommend limits level for high risk", func(t *testing.T) {
		ctx := TransactionContext{
			ValueUSD:  1000.0,
			RiskScore: 0.8,
		}
		rec := engine.Recommend(ctx)
		if rec.Level > PrivacyConfidential {
			t.Errorf("High risk recommendation = %v, want <= Confidential", rec.Level)
		}
	})

	t.Run("CanMorph allows increasing privacy", func(t *testing.T) {
		can, _ := engine.CanMorph(PrivacyPrivate, PrivacySovereign)
		if !can {
			t.Error("CanMorph should allow increasing privacy")
		}
	})

	t.Run("GetMorphPath returns single step for increase", func(t *testing.T) {
		path := engine.GetMorphPath(PrivacyConfidential, PrivacySovereign)
		if len(path) != 1 || path[0] != PrivacySovereign {
			t.Errorf("Increase path = %v, want [Sovereign]", path)
		}
	})

	t.Run("GetMorphPath returns incremental steps for decrease", func(t *testing.T) {
		path := engine.GetMorphPath(PrivacySovereign, PrivacyConfidential)
		if len(path) != 3 {
			t.Errorf("Decrease path length = %d, want 3", len(path))
		}
	})
}

func TestLocalProofGenerator(t *testing.T) {
	generator := NewLocalProofGenerator()

	t.Run("Generate produces valid proof", func(t *testing.T) {
		data := []byte("test circuit data")
		result, err := generator.Generate(data, PrivacyPrivate)
		if err != nil {
			t.Fatalf("Generate returned error: %v", err)
		}

		if result.ProofID == "" {
			t.Error("ProofID should not be empty")
		}
		if len(result.ProofData) == 0 {
			t.Error("ProofData should not be empty")
		}
		if result.PrivacyLevel != int(PrivacyPrivate) {
			t.Errorf("PrivacyLevel = %d, want %d", result.PrivacyLevel, PrivacyPrivate)
		}
		if result.QualityScore <= 0 {
			t.Error("QualityScore should be positive")
		}
	})

	t.Run("Higher levels produce larger proofs", func(t *testing.T) {
		data := []byte("test data")
		var prevSize int
		for level := PrivacyTransparent; level <= PrivacySovereign; level++ {
			result, err := generator.Generate(data, level)
			if err != nil {
				t.Fatalf("Generate(%v) returned error: %v", level, err)
			}
			size := len(result.ProofData)
			if size < prevSize {
				t.Errorf("Proof size for level %d (%d) < previous (%d)", level, size, prevSize)
			}
			prevSize = size
		}
	})
}

func TestLocalProofVerifier(t *testing.T) {
	verifier := NewLocalProofVerifier()

	t.Run("Verify accepts valid proof", func(t *testing.T) {
		proof := make([]byte, 64)
		level := PrivacyPseudonymous
		if !verifier.Verify(proof, &level) {
			t.Error("Verify should accept valid proof")
		}
	})

	t.Run("Verify rejects too-short proof", func(t *testing.T) {
		proof := make([]byte, 16)
		if verifier.Verify(proof, nil) {
			t.Error("Verify should reject proof < 32 bytes")
		}
	})

	t.Run("Verify checks size against privacy level", func(t *testing.T) {
		proof := make([]byte, 32)
		level := PrivacyAnonymous
		if verifier.Verify(proof, &level) {
			t.Error("Verify should reject proof too small for Anonymous level")
		}
	})
}

func TestCrossChainBridge(t *testing.T) {
	bridge := NewCrossChainBridge()

	t.Run("IsRouteSupported returns true for valid routes", func(t *testing.T) {
		if !bridge.IsRouteSupported("ethereum", "polygon") {
			t.Error("ethereum -> polygon should be supported")
		}
		if !bridge.IsRouteSupported("polygon", "ethereum") {
			t.Error("polygon -> ethereum should be supported")
		}
	})

	t.Run("IsRouteSupported returns false for invalid routes", func(t *testing.T) {
		if bridge.IsRouteSupported("bitcoin", "solana") {
			t.Error("bitcoin -> solana should not be supported")
		}
	})

	t.Run("GetQuote returns valid quote", func(t *testing.T) {
		quote, err := bridge.GetQuote("ethereum", "polygon", 1000000, PrivacyPrivate)
		if err != nil {
			t.Fatalf("GetQuote returned error: %v", err)
		}

		if quote.SourceChain != "ethereum" {
			t.Errorf("SourceChain = %s, want ethereum", quote.SourceChain)
		}
		if quote.TotalFee == 0 {
			t.Error("TotalFee should not be zero")
		}
		if quote.EstimatedTimeSeconds == 0 {
			t.Error("EstimatedTimeSeconds should not be zero")
		}
	})

	t.Run("GetQuote returns error for unsupported route", func(t *testing.T) {
		_, err := bridge.GetQuote("bitcoin", "solana", 1000000, PrivacyPrivate)
		if err == nil {
			t.Error("GetQuote should return error for unsupported route")
		}
	})

	t.Run("Higher privacy levels cost more", func(t *testing.T) {
		quoteLow, _ := bridge.GetQuote("ethereum", "polygon", 1000000, PrivacyPseudonymous)
		quoteHigh, _ := bridge.GetQuote("ethereum", "polygon", 1000000, PrivacySovereign)

		if quoteHigh.FeeSourceChain <= quoteLow.FeeSourceChain {
			t.Error("Higher privacy should cost more")
		}
	})
}

func TestComplianceProver(t *testing.T) {
	prover := NewComplianceProver()

	t.Run("ProveAge generates valid proof", func(t *testing.T) {
		encrypted := make([]byte, 32)
		proof, err := prover.ProveAge(encrypted, 18)
		if err != nil {
			t.Fatalf("ProveAge returned error: %v", err)
		}

		if proof.ProofType != ComplianceAgeVerification {
			t.Errorf("ProofType = %s, want %s", proof.ProofType, ComplianceAgeVerification)
		}
		if !proof.Verified {
			t.Error("Proof should be verified")
		}
		if proof.Metadata["minimum_age"] != 18 {
			t.Errorf("minimum_age = %v, want 18", proof.Metadata["minimum_age"])
		}
	})

	t.Run("ProveAccreditedInvestor generates valid proof", func(t *testing.T) {
		encrypted := make([]byte, 32)
		proof, err := prover.ProveAccreditedInvestor(encrypted, encrypted, "US")
		if err != nil {
			t.Fatalf("ProveAccreditedInvestor returned error: %v", err)
		}

		if proof.ProofType != ComplianceAccreditedInvestor {
			t.Errorf("ProofType = %s, want %s", proof.ProofType, ComplianceAccreditedInvestor)
		}
		if proof.Metadata["jurisdiction"] != "US" {
			t.Errorf("jurisdiction = %v, want US", proof.Metadata["jurisdiction"])
		}
	})

	t.Run("ProveNotSanctioned generates valid proof", func(t *testing.T) {
		identity := make([]byte, 32)
		list := make([]byte, 32)
		proof, err := prover.ProveNotSanctioned(identity, list)
		if err != nil {
			t.Fatalf("ProveNotSanctioned returned error: %v", err)
		}

		if proof.ProofType != ComplianceSanctions {
			t.Errorf("ProofType = %s, want %s", proof.ProofType, ComplianceSanctions)
		}
	})

	t.Run("ProveTransactionUnderLimit generates valid proof", func(t *testing.T) {
		encrypted := make([]byte, 32)
		proof, err := prover.ProveTransactionUnderLimit(encrypted, 10000.0)
		if err != nil {
			t.Fatalf("ProveTransactionUnderLimit returned error: %v", err)
		}

		if proof.ProofType != ComplianceTransactionLimit {
			t.Errorf("ProofType = %s, want %s", proof.ProofType, ComplianceTransactionLimit)
		}
		if proof.Metadata["limit_usd"] != 10000.0 {
			t.Errorf("limit_usd = %v, want 10000.0", proof.Metadata["limit_usd"])
		}
	})

	t.Run("GetJurisdictionProfile returns valid profiles", func(t *testing.T) {
		us := prover.GetJurisdictionProfile("US")
		if !us.RequiresKYC {
			t.Error("US should require KYC")
		}
		if us.MaxPrivacyLevel != 4 {
			t.Errorf("US MaxPrivacyLevel = %d, want 4", us.MaxPrivacyLevel)
		}

		eu := prover.GetJurisdictionProfile("EU")
		if eu.MaxPrivacyLevel != 5 {
			t.Errorf("EU MaxPrivacyLevel = %d, want 5", eu.MaxPrivacyLevel)
		}
	})

	t.Run("GetJurisdictionProfile returns default for unknown", func(t *testing.T) {
		unknown := prover.GetJurisdictionProfile("XX")
		if unknown.MaxPrivacyLevel != 3 {
			t.Errorf("Unknown jurisdiction MaxPrivacyLevel = %d, want 3", unknown.MaxPrivacyLevel)
		}
	})
}
