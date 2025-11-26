# NexusZero Python SDK

High-level Python interface for the NexusZero Protocol - privacy-preserving transactions with quantum-resistant zero-knowledge proofs.

## Installation

```bash
pip install nexuszero
```

Or from source:

```bash
cd sdks/python
pip install -e ".[dev]"
```

## Quick Start

```python
import asyncio
from nexuszero import NexusZeroClient, PrivacyLevel

async def main():
    async with NexusZeroClient(api_key="your-api-key") as client:
        # Create a privacy-preserving transaction
        tx = await client.create_transaction(
            recipient="0x...",
            amount=1000000000000000000,  # 1 ETH in wei
            privacy_level=PrivacyLevel.PRIVATE,
            chain="ethereum",
        )
        print(f"Transaction ID: {tx.id}")

        # Generate a ZK proof
        proof = await client.generate_proof(
            data=b"sensitive data",
            privacy_level=PrivacyLevel.ANONYMOUS,
        )
        print(f"Proof ID: {proof.proof_id}")

        # Get privacy recommendation
        rec = client.recommend_privacy(
            value_usd=50000.0,
            requires_compliance=True,
        )
        print(f"Recommended: {rec.level.name} - {rec.level.description}")

asyncio.run(main())
```

## Privacy Levels

NexusZero implements a 6-level privacy spectrum:

| Level | Name         | Description                       |
| ----- | ------------ | --------------------------------- |
| 0     | Transparent  | Public blockchain parity          |
| 1     | Pseudonymous | Address obfuscation               |
| 2     | Confidential | Encrypted amounts                 |
| 3     | Private      | Full transaction privacy          |
| 4     | Anonymous    | Unlinkable transactions           |
| 5     | Sovereign    | Maximum quantum-resistant privacy |

## Local Operations

Some operations can be performed locally without API calls:

```python
from nexuszero import PrivacyEngine, ProofGenerator, ProofVerifier

# Get privacy recommendations locally
engine = PrivacyEngine()
rec = engine.recommend(TransactionContext(value_usd=10000.0))

# Generate proofs locally (for testing)
generator = ProofGenerator()
result = generator.generate(b"data", PrivacyLevel.PRIVATE)

# Verify proofs locally
verifier = ProofVerifier()
is_valid = verifier.verify(result.proof_data)
```

## Cross-Chain Bridge

```python
from nexuszero import CrossChainBridge

bridge = CrossChainBridge()

# Check if route is supported
if bridge.is_route_supported("ethereum", "polygon"):
    # Get quote
    quote = bridge.get_quote("ethereum", "polygon", 1000000)
    print(f"Fee: {quote.total_fee}, Time: {quote.estimated_time_seconds}s")
```

## Compliance Proofs

Generate ZK proofs for regulatory compliance without revealing sensitive data:

```python
from nexuszero import ComplianceProver

prover = ComplianceProver()

# Prove age without revealing actual birthdate
age_proof = prover.prove_age(
    encrypted_birthdate=encrypted_data,
    minimum_age=18,
)

# Prove accredited investor status
investor_proof = prover.prove_accredited_investor(
    encrypted_net_worth=encrypted_worth,
    encrypted_income=encrypted_income,
    jurisdiction="US",
)

# Prove not on sanctions list
sanctions_proof = prover.prove_not_sanctioned(
    encrypted_identity_hash=id_hash,
    sanctions_list_hash=list_hash,
)
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Type checking
mypy nexuszero

# Linting
ruff check nexuszero
```

## License

AGPL-3.0-or-later
