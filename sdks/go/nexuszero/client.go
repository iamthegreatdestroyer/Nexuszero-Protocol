// Package nexuszero provides a Go client for the NexusZero Protocol.
//
// NexusZero is a quantum-resistant zero-knowledge proof privacy infrastructure
// that enables privacy-preserving transactions across multiple blockchains.
//
// # Privacy Levels
//
// NexusZero implements a 6-level privacy spectrum:
//
//	Level 0 (Transparent):   Public blockchain parity
//	Level 1 (Pseudonymous):  Address obfuscation
//	Level 2 (Confidential):  Encrypted amounts
//	Level 3 (Private):       Full transaction privacy
//	Level 4 (Anonymous):     Unlinkable transactions
//	Level 5 (Sovereign):     Maximum quantum-resistant privacy
//
// # Quick Start
//
//	client := nexuszero.NewClient("https://api.nexuszero.io", "your-api-key")
//
//	// Create a transaction
//	tx, err := client.CreateTransaction(ctx, &nexuszero.TransactionRequest{
//	    Recipient:    "0x...",
//	    Amount:       1000000000000000000, // 1 ETH in wei
//	    PrivacyLevel: nexuszero.PrivacyPrivate,
//	    Chain:        "ethereum",
//	})
//
//	// Generate a proof
//	proof, err := client.GenerateProof(ctx, data, nexuszero.PrivacyAnonymous)
package nexuszero

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/google/uuid"
)

// PrivacyLevel represents the 6-level privacy spectrum.
type PrivacyLevel uint8

const (
	// PrivacyTransparent - Level 0: Public blockchain parity.
	PrivacyTransparent PrivacyLevel = 0
	// PrivacyPseudonymous - Level 1: Address obfuscation.
	PrivacyPseudonymous PrivacyLevel = 1
	// PrivacyConfidential - Level 2: Encrypted amounts.
	PrivacyConfidential PrivacyLevel = 2
	// PrivacyPrivate - Level 3: Full transaction privacy.
	PrivacyPrivate PrivacyLevel = 3
	// PrivacyAnonymous - Level 4: Unlinkable transactions.
	PrivacyAnonymous PrivacyLevel = 4
	// PrivacySovereign - Level 5: Maximum quantum-resistant privacy.
	PrivacySovereign PrivacyLevel = 5
)

// String returns a human-readable name for the privacy level.
func (p PrivacyLevel) String() string {
	names := []string{"Transparent", "Pseudonymous", "Confidential", "Private", "Anonymous", "Sovereign"}
	if int(p) < len(names) {
		return names[p]
	}
	return fmt.Sprintf("Unknown(%d)", p)
}

// SecurityBits returns the security level in bits for this privacy level.
func (p PrivacyLevel) SecurityBits() int {
	bits := []int{0, 80, 128, 192, 256, 256}
	if int(p) < len(bits) {
		return bits[p]
	}
	return 0
}

// EstimatedProofTimeMs returns the estimated proof generation time in milliseconds.
func (p PrivacyLevel) EstimatedProofTimeMs() int {
	times := []int{0, 50, 100, 250, 500, 1000}
	if int(p) < len(times) {
		return times[p]
	}
	return 0
}

// TransactionStatus represents the lifecycle state of a transaction.
type TransactionStatus int

const (
	StatusCreated TransactionStatus = iota
	StatusPrivacySelected
	StatusProofGenerating
	StatusProofGenerated
	StatusSubmitted
	StatusConfirmed
	StatusFailed
)

// String returns a human-readable name for the status.
func (s TransactionStatus) String() string {
	names := []string{"Created", "PrivacySelected", "ProofGenerating", "ProofGenerated", "Submitted", "Confirmed", "Failed"}
	if int(s) < len(names) {
		return names[s]
	}
	return fmt.Sprintf("Unknown(%d)", s)
}

// TransactionRequest represents a request to create a transaction.
type TransactionRequest struct {
	Recipient    string            `json:"recipient"`
	Amount       uint64            `json:"amount"`
	PrivacyLevel PrivacyLevel      `json:"privacy_level"`
	Chain        string            `json:"chain"`
	Metadata     map[string]any    `json:"metadata,omitempty"`
}

// Transaction represents a privacy-preserving transaction.
type Transaction struct {
	ID                   string            `json:"id"`
	SenderCommitment     []byte            `json:"sender_commitment"`
	RecipientCommitment  []byte            `json:"recipient_commitment"`
	AmountCommitment     []byte            `json:"amount_commitment,omitempty"`
	PrivacyLevel         PrivacyLevel      `json:"privacy_level"`
	ProofID              string            `json:"proof_id,omitempty"`
	Chain                string            `json:"chain"`
	ChainTxHash          string            `json:"chain_tx_hash,omitempty"`
	Status               TransactionStatus `json:"status"`
	Metadata             map[string]any    `json:"metadata"`
	CreatedAt            time.Time         `json:"created_at"`
	UpdatedAt            time.Time         `json:"updated_at"`
}

// ProofResult represents the result of proof generation.
type ProofResult struct {
	ProofID          string  `json:"proof_id"`
	ProofData        []byte  `json:"proof_data"`
	PrivacyLevel     int     `json:"privacy_level"`
	GenerationTimeMs int     `json:"generation_time_ms"`
	QualityScore     float64 `json:"quality_score"`
	Verified         bool    `json:"verified"`
}

// FeeEstimate represents gas/fee estimates for operations.
type FeeEstimate struct {
	GasUnits       uint64  `json:"gas_units"`
	GasPriceGwei   float64 `json:"gas_price_gwei"`
	TotalFeeNative float64 `json:"total_fee_native"`
	TotalFeeUSD    float64 `json:"total_fee_usd"`
}

// Client is the main NexusZero Protocol client.
type Client struct {
	apiURL     string
	apiKey     string
	httpClient *http.Client
}

// ClientOption configures a Client.
type ClientOption func(*Client)

// WithHTTPClient sets a custom HTTP client.
func WithHTTPClient(httpClient *http.Client) ClientOption {
	return func(c *Client) {
		c.httpClient = httpClient
	}
}

// WithTimeout sets the HTTP client timeout.
func WithTimeout(timeout time.Duration) ClientOption {
	return func(c *Client) {
		c.httpClient.Timeout = timeout
	}
}

// NewClient creates a new NexusZero client.
func NewClient(apiURL, apiKey string, opts ...ClientOption) *Client {
	c := &Client{
		apiURL: apiURL,
		apiKey: apiKey,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}

	for _, opt := range opts {
		opt(c)
	}

	return c
}

// doRequest performs an HTTP request with authentication.
func (c *Client) doRequest(ctx context.Context, method, path string, body any) (*http.Response, error) {
	var bodyReader io.Reader
	if body != nil {
		jsonBody, err := json.Marshal(body)
		if err != nil {
			return nil, fmt.Errorf("marshal request body: %w", err)
		}
		bodyReader = bytes.NewReader(jsonBody)
	}

	req, err := http.NewRequestWithContext(ctx, method, c.apiURL+path, bodyReader)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}

	return c.httpClient.Do(req)
}

// parseResponse parses a JSON response into the target struct.
func parseResponse[T any](resp *http.Response) (*T, error) {
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
	}

	var result T
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	return &result, nil
}

// CreateTransaction creates a new privacy-preserving transaction.
func (c *Client) CreateTransaction(ctx context.Context, req *TransactionRequest) (*Transaction, error) {
	resp, err := c.doRequest(ctx, http.MethodPost, "/api/v1/transactions", req)
	if err != nil {
		return nil, err
	}
	return parseResponse[Transaction](resp)
}

// GetTransaction retrieves a transaction by ID.
func (c *Client) GetTransaction(ctx context.Context, txID string) (*Transaction, error) {
	resp, err := c.doRequest(ctx, http.MethodGet, "/api/v1/transactions/"+txID, nil)
	if err != nil {
		return nil, err
	}
	return parseResponse[Transaction](resp)
}

// GenerateProof generates a zero-knowledge proof.
func (c *Client) GenerateProof(ctx context.Context, data []byte, level PrivacyLevel) (*ProofResult, error) {
	body := map[string]any{
		"data":          fmt.Sprintf("%x", data),
		"privacy_level": int(level),
	}

	resp, err := c.doRequest(ctx, http.MethodPost, "/api/v1/proofs/generate", body)
	if err != nil {
		return nil, err
	}
	return parseResponse[ProofResult](resp)
}

// VerifyProof verifies a zero-knowledge proof.
func (c *Client) VerifyProof(ctx context.Context, proof []byte) (bool, error) {
	body := map[string]any{
		"proof": fmt.Sprintf("%x", proof),
	}

	resp, err := c.doRequest(ctx, http.MethodPost, "/api/v1/proofs/verify", body)
	if err != nil {
		return false, err
	}

	result, err := parseResponse[map[string]any](resp)
	if err != nil {
		return false, err
	}

	valid, ok := (*result)["valid"].(bool)
	return ok && valid, nil
}

// GetBridgeQuote gets a quote for cross-chain bridging.
func (c *Client) GetBridgeQuote(ctx context.Context, fromChain, toChain string, amount uint64, level PrivacyLevel) (*FeeEstimate, error) {
	body := map[string]any{
		"from_chain":    fromChain,
		"to_chain":      toChain,
		"amount":        amount,
		"privacy_level": int(level),
	}

	resp, err := c.doRequest(ctx, http.MethodPost, "/api/v1/bridge/quote", body)
	if err != nil {
		return nil, err
	}
	return parseResponse[FeeEstimate](resp)
}

// InitiateBridge initiates a cross-chain bridge transfer.
func (c *Client) InitiateBridge(ctx context.Context, fromChain, toChain string, amount uint64, recipient string, level PrivacyLevel) (string, error) {
	body := map[string]any{
		"from_chain":    fromChain,
		"to_chain":      toChain,
		"amount":        amount,
		"recipient":     recipient,
		"privacy_level": int(level),
	}

	resp, err := c.doRequest(ctx, http.MethodPost, "/api/v1/bridge/initiate", body)
	if err != nil {
		return "", err
	}

	result, err := parseResponse[map[string]any](resp)
	if err != nil {
		return "", err
	}

	transferID, ok := (*result)["transfer_id"].(string)
	if !ok {
		return "", fmt.Errorf("invalid response: missing transfer_id")
	}

	return transferID, nil
}

// HealthCheck performs a health check on the API.
func (c *Client) HealthCheck(ctx context.Context) (bool, error) {
	resp, err := c.doRequest(ctx, http.MethodGet, "/health", nil)
	if err != nil {
		return false, err
	}
	defer resp.Body.Close()

	return resp.StatusCode == http.StatusOK, nil
}
