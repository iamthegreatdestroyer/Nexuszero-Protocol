package nexuszero

import (
	"fmt"
)

// BridgeStatus represents the status of a bridge transfer.
type BridgeStatus int

const (
	BridgeStatusInitiated BridgeStatus = iota
	BridgeStatusProofGenerated
	BridgeStatusSourceConfirmed
	BridgeStatusRelaying
	BridgeStatusTargetPending
	BridgeStatusCompleted
	BridgeStatusFailed
)

// String returns a human-readable name for the status.
func (s BridgeStatus) String() string {
	names := []string{"Initiated", "ProofGenerated", "SourceConfirmed", "Relaying", "TargetPending", "Completed", "Failed"}
	if int(s) < len(names) {
		return names[s]
	}
	return fmt.Sprintf("Unknown(%d)", s)
}

// BridgeQuote represents a quote for a bridge transfer.
type BridgeQuote struct {
	SourceChain          string  `json:"source_chain"`
	TargetChain          string  `json:"target_chain"`
	Amount               uint64  `json:"amount"`
	FeeSourceChain       uint64  `json:"fee_source_chain"`
	FeeTargetChain       uint64  `json:"fee_target_chain"`
	FeeProtocol          uint64  `json:"fee_protocol"`
	TotalFee             uint64  `json:"total_fee"`
	EstimatedTimeSeconds int     `json:"estimated_time_seconds"`
	ExchangeRate         float64 `json:"exchange_rate"`
}

// CrossChainBridge provides bridge functionality.
type CrossChainBridge struct {
	supportedRoutes map[string]bool
	baseFees        map[string]uint64
	timeEstimates   map[string]int
}

// NewCrossChainBridge creates a new bridge instance.
func NewCrossChainBridge() *CrossChainBridge {
	routes := make(map[string]bool)
	routePairs := [][2]string{
		{"ethereum", "polygon"},
		{"ethereum", "arbitrum"},
		{"ethereum", "optimism"},
		{"ethereum", "base"},
		{"polygon", "ethereum"},
		{"arbitrum", "ethereum"},
		{"optimism", "ethereum"},
		{"base", "ethereum"},
	}
	for _, pair := range routePairs {
		routes[pair[0]+"-"+pair[1]] = true
	}

	return &CrossChainBridge{
		supportedRoutes: routes,
		baseFees: map[string]uint64{
			"ethereum": 50000,
			"polygon":  1000,
			"arbitrum": 5000,
			"optimism": 5000,
			"base":     3000,
		},
		timeEstimates: map[string]int{
			"ethereum": 900,
			"polygon":  120,
			"arbitrum": 600,
			"optimism": 600,
			"base":     120,
		},
	}
}

// IsRouteSupported checks if a bridge route is supported.
func (b *CrossChainBridge) IsRouteSupported(sourceChain, targetChain string) bool {
	return b.supportedRoutes[sourceChain+"-"+targetChain]
}

// GetQuote gets a quote for a bridge transfer.
func (b *CrossChainBridge) GetQuote(sourceChain, targetChain string, amount uint64, privacyLevel PrivacyLevel) (*BridgeQuote, error) {
	if !b.IsRouteSupported(sourceChain, targetChain) {
		return nil, fmt.Errorf("route %s -> %s is not supported", sourceChain, targetChain)
	}

	// Privacy level multiplier
	privacyMultiplier := 1.0 + float64(privacyLevel)*0.2

	sourceFee := uint64(float64(b.baseFees[sourceChain]) * privacyMultiplier)
	targetFee := uint64(float64(b.baseFees[targetChain]) * privacyMultiplier)
	protocolFee := amount / 1000 // 0.1%

	// Get the longer of the two chain times
	sourceTime := b.timeEstimates[sourceChain]
	targetTime := b.timeEstimates[targetChain]
	estTime := sourceTime
	if targetTime > estTime {
		estTime = targetTime
	}

	return &BridgeQuote{
		SourceChain:          sourceChain,
		TargetChain:          targetChain,
		Amount:               amount,
		FeeSourceChain:       sourceFee,
		FeeTargetChain:       targetFee,
		FeeProtocol:          protocolFee,
		TotalFee:             sourceFee + targetFee + protocolFee,
		EstimatedTimeSeconds: estTime,
		ExchangeRate:         1.0,
	}, nil
}

// EstimateTime estimates the bridge completion time in seconds.
func (b *CrossChainBridge) EstimateTime(sourceChain, targetChain string) int {
	sourceTime := b.timeEstimates[sourceChain]
	targetTime := b.timeEstimates[targetChain]
	if targetTime > sourceTime {
		return targetTime
	}
	return sourceTime
}

// ListSupportedChains returns a list of supported chains.
func (b *CrossChainBridge) ListSupportedChains() []string {
	chains := make(map[string]bool)
	for route := range b.supportedRoutes {
		// Parse "source-target" format
		for i, c := range route {
			if c == '-' {
				chains[route[:i]] = true
				chains[route[i+1:]] = true
				break
			}
		}
	}

	result := make([]string, 0, len(chains))
	for chain := range chains {
		result = append(result, chain)
	}
	return result
}
