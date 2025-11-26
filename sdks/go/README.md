# NexusZero Go SDK

Go client library for the NexusZero Protocol - privacy-preserving transactions with quantum-resistant zero-knowledge proofs.

## Installation

```bash
go get github.com/iamthegreatdestroyer/Nexuszero-Protocol/sdks/go
```

## Quick Start

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/iamthegreatdestroyer/Nexuszero-Protocol/sdks/go/nexuszero"
)

func main() {
    // Create client
    client := nexuszero.NewClient("https://api.nexuszero.io", "your-api-key")

    ctx := context.Background()

    // Create a privacy-preserving transaction
    tx, err := client.CreateTransaction(ctx, &nexuszero.TransactionRequest{
        Recipient:    "0x...",
        Amount:       1000000000000000000, // 1 ETH in wei
        PrivacyLevel: nexuszero.PrivacyPrivate,
        Chain:        "ethereum",
    })
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Transaction ID: %s\n", tx.ID)

    // Generate a proof
    proof, err := client.GenerateProof(ctx, []byte("sensitive data"), nexuszero.PrivacyAnonymous)
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Proof ID: %s\n", proof.ProofID)
}
```

## Privacy Levels

NexusZero implements a 6-level privacy spectrum:

| Level | Constant              | Description                       |
| ----- | --------------------- | --------------------------------- |
| 0     | `PrivacyTransparent`  | Public blockchain parity          |
| 1     | `PrivacyPseudonymous` | Address obfuscation               |
| 2     | `PrivacyConfidential` | Encrypted amounts                 |
| 3     | `PrivacyPrivate`      | Full transaction privacy          |
| 4     | `PrivacyAnonymous`    | Unlinkable transactions           |
| 5     | `PrivacySovereign`    | Maximum quantum-resistant privacy |

## Local Operations

Some operations can be performed locally without API calls:

```go
// Privacy Engine for recommendations
engine := nexuszero.NewPrivacyEngine()
rec := engine.Recommend(nexuszero.TransactionContext{
    ValueUSD: 10000.0,
    RequiresCompliance: false,
})
fmt.Printf("Recommended: %s\n", rec.Level)

// Local proof generation (for testing)
generator := nexuszero.NewLocalProofGenerator()
result, _ := generator.Generate([]byte("data"), nexuszero.PrivacyPrivate)

// Local proof verification
verifier := nexuszero.NewLocalProofVerifier()
isValid := verifier.Verify(result.ProofData, nil)
```

## Cross-Chain Bridge

```go
bridge := nexuszero.NewCrossChainBridge()

// Check if route is supported
if bridge.IsRouteSupported("ethereum", "polygon") {
    // Get quote
    quote, _ := bridge.GetQuote("ethereum", "polygon", 1000000, nexuszero.PrivacyPrivate)
    fmt.Printf("Fee: %d, Time: %ds\n", quote.TotalFee, quote.EstimatedTimeSeconds)
}
```

## Compliance Proofs

Generate ZK proofs for regulatory compliance:

```go
prover := nexuszero.NewComplianceProver()

// Prove age without revealing birthdate
ageProof, _ := prover.ProveAge(encryptedBirthdate, 18)

// Prove accredited investor status
investorProof, _ := prover.ProveAccreditedInvestor(encryptedNetWorth, encryptedIncome, "US")

// Prove not on sanctions list
sanctionsProof, _ := prover.ProveNotSanctioned(identityHash, listHash)
```

## Testing

```bash
cd sdks/go
go test ./...
```

## License

AGPL-3.0-or-later
