# NexusZero Protocol - Phase 1 Implementation

## 🚀 Overview

Phase 1 establishes the foundational microservices architecture for NexusZero Protocol, implementing the core transaction pipeline with privacy and compliance capabilities.

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         NEXUSZERO PROTOCOL - PHASE 1                        │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        API GATEWAY (8080)                           │   │
│  │  • JWT Authentication    • Rate Limiting     • Circuit Breaker      │   │
│  │  • Request Routing       • Load Balancing    • WebSocket Support    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│         ┌──────────────────────────┼──────────────────────────┐            │
│         │                          │                          │            │
│         ▼                          ▼                          ▼            │
│  ┌─────────────┐          ┌─────────────┐          ┌─────────────┐        │
│  │ TRANSACTION │          │   PRIVACY   │          │ COMPLIANCE  │        │
│  │   SERVICE   │◄────────►│   SERVICE   │◄────────►│   SERVICE   │        │
│  │   (8081)    │          │   (8082)    │          │   (8083)    │        │
│  │             │          │             │          │             │        │
│  │ • Mempool   │          │ • ZKP Gen   │          │ • AML/KYC   │        │
│  │ • Validate  │          │ • APM Engine│          │ • Sanctions │        │
│  │ • Process   │          │ • Morphing  │          │ • Rules     │        │
│  └─────────────┘          └─────────────┘          └─────────────┘        │
│         │                          │                          │            │
│         │                          │                          │            │
│         └──────────────────────────┼──────────────────────────┘            │
│                                    │                                        │
│                                    ▼                                        │
│                        ┌─────────────────────┐                             │
│                        │   BRIDGE SERVICE    │                             │
│                        │      (8084)         │                             │
│                        │                     │                             │
│                        │ • HTLC Atomic Swaps │                             │
│                        │ • Multi-Chain       │                             │
│                        │ • Liquidity Pools   │                             │
│                        └─────────────────────┘                             │
│                                    │                                        │
│         ┌──────────────────────────┼──────────────────────────┐            │
│         │                          │                          │            │
│         ▼                          ▼                          ▼            │
│  ┌─────────────┐          ┌─────────────┐          ┌─────────────┐        │
│  │  Ethereum   │          │   Polygon   │          │  Arbitrum   │        │
│  │   (EVM)     │          │   (EVM)     │          │   (EVM)     │        │
│  └─────────────┘          └─────────────┘          └─────────────┘        │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Components Implemented

### 1. Shared Libraries (`shared/`)

| Library                | Purpose                                                         |
| ---------------------- | --------------------------------------------------------------- |
| `nexuszero-common`     | Common types, errors, utilities, configuration                  |
| `nexuszero-crypto-lib` | Cryptographic primitives, ZKP foundations, Pedersen commitments |
| `nexuszero-db`         | Database connection pooling, migrations, Redis client           |

### 2. API Gateway (`services/api_gateway/`)

**Port:** 8080

**Features:**

- JWT-based authentication
- Rate limiting (Redis-backed sliding window)
- Circuit breaker pattern for downstream services
- Request routing with health checks
- CORS and security middleware
- WebSocket support for real-time updates
- Prometheus metrics endpoint

**Endpoints:**

```
POST   /v1/auth/register        - User registration
POST   /v1/auth/login           - User login
POST   /v1/auth/refresh         - Token refresh

GET    /v1/transactions         - List transactions
POST   /v1/transactions         - Create transaction
GET    /v1/transactions/:id     - Get transaction details

POST   /v1/privacy/proof        - Generate privacy proof
POST   /v1/privacy/verify       - Verify privacy proof

POST   /v1/compliance/check     - Run compliance check
GET    /v1/compliance/status    - Get compliance status

GET    /v1/bridge/chains        - List supported chains
GET    /v1/bridge/routes        - Get bridge routes
POST   /v1/bridge/quote         - Get bridge quote
POST   /v1/bridge/transfer      - Initiate bridge transfer

GET    /health                  - Health check
GET    /metrics                 - Prometheus metrics
WS     /ws                      - WebSocket connection
```

### 3. Transaction Service (`services/transaction_service/`)

**Port:** 8081

**Features:**

- NTP (NexusZero Transaction Protocol) implementation
- Redis-backed mempool for pending transactions
- Transaction validation pipeline
- Multi-signature support
- Fee estimation and calculation
- Transaction batching for efficiency
- Event streaming for real-time updates

**Transaction Flow:**

```
Submit → Validate → Privacy Check → Compliance Check → Mempool → Process → Confirm
```

### 4. Privacy Service (`services/privacy_service/`)

**Port:** 8082

**Features:**

- APM (Adaptive Privacy Morphing) engine
- Zero-Knowledge Proof generation
- Range proofs using Bulletproofs
- Pedersen commitments for hidden values
- Privacy level adaptation based on context
- Proof caching for performance
- Batch proof verification

**Privacy Levels:**
| Level | Description | ZKP Type |
|-------|-------------|----------|
| Standard | Basic transaction privacy | Pedersen commitment |
| Enhanced | Additional metadata hiding | Range proof |
| Maximum | Full anonymity set | Ring signature (Phase 2) |

### 5. Compliance Service (`services/compliance_service/`)

**Port:** 8083

**Features:**

- RCL (Regulatory Compliance Layer) implementation
- Jurisdiction-based rule engine (Rhai scripting)
- AML/KYC verification integration
- Sanctions list screening (OFAC, EU, UN)
- Transaction monitoring
- Suspicious Activity Report (SAR) generation
- Audit trail logging

**Supported Jurisdictions:**

- United States (FINCEN/SEC)
- European Union (MiCA)
- United Kingdom (FCA)
- Singapore (MAS)
- Switzerland (FINMA)
- And 40+ more...

### 6. Bridge Service (`services/bridge_service/`)

**Port:** 8084

**Features:**

- Cross-chain atomic swaps using HTLC
- Multi-chain support (EVM compatible)
- Liquidity pool management
- Quote generation with fee estimation
- Relayer network coordination
- Transfer tracking and retry logic
- Real-time event monitoring

**Supported Chains:**

- Ethereum (mainnet, testnets)
- Polygon
- Arbitrum One
- Optimism
- Base
- BNB Smart Chain
- Avalanche C-Chain
- Solana (Phase 2)

---

## 🛠️ Technology Stack

### Core Technologies

| Category      | Technology | Version |
| ------------- | ---------- | ------- |
| Language      | Rust       | 1.82+   |
| Web Framework | Axum       | 0.7     |
| Database      | PostgreSQL | 16      |
| Cache         | Redis      | 7       |
| Container     | Docker     | 24+     |

### Cryptography

| Purpose         | Library          |
| --------------- | ---------------- |
| Elliptic Curves | curve25519-dalek |
| Signatures      | ed25519-dalek    |
| ZKP Framework   | Bulletproofs     |
| Hashing         | SHA-256, BLAKE3  |
| Commitments     | Pedersen         |

### Observability

| Purpose       | Tool                    |
| ------------- | ----------------------- |
| Metrics       | Prometheus              |
| Visualization | Grafana                 |
| Logging       | Tracing (structured)    |
| Tracing       | OpenTelemetry (Phase 2) |

---

## 🚀 Quick Start

### Prerequisites

- Rust 1.82+
- Docker & Docker Compose
- PostgreSQL 16 (or use Docker)
- Redis 7 (or use Docker)

### Development Setup

```bash
# Clone the repository
git clone https://github.com/iamthegreatdestroyer/Nexuszero-Protocol.git
cd Nexuszero-Protocol

# Start infrastructure (PostgreSQL, Redis)
docker-compose -f services/docker-compose.services.yml up -d postgres redis

# Run database migrations
sqlx database create
sqlx migrate run

# Build all services
cargo build --workspace

# Run individual services (in separate terminals)
cargo run --package api_gateway
cargo run --package transaction_service
cargo run --package privacy_service
cargo run --package compliance_service
cargo run --package bridge_service
```

### Docker Deployment

```bash
# Build and start all services
docker-compose -f services/docker-compose.services.yml up -d --build

# Check service health
curl http://localhost:8080/health

# View logs
docker-compose -f services/docker-compose.services.yml logs -f

# Stop all services
docker-compose -f services/docker-compose.services.yml down
```

---

## 📊 Configuration

### Environment Variables

```bash
# Database
DATABASE_URL=postgres://nexuszero:password@localhost:5432/nexuszero

# Redis
REDIS_URL=redis://localhost:6379

# JWT
JWT_SECRET=your-secret-key-here

# Service URLs (for inter-service communication)
TRANSACTION_SERVICE_URL=http://localhost:8081
PRIVACY_SERVICE_URL=http://localhost:8082
COMPLIANCE_SERVICE_URL=http://localhost:8083
BRIDGE_SERVICE_URL=http://localhost:8084

# Chain RPC (for Bridge Service)
ETHEREUM_RPC_URL=https://eth-mainnet.g.alchemy.com/v2/your-key
POLYGON_RPC_URL=https://polygon-mainnet.g.alchemy.com/v2/your-key
ARBITRUM_RPC_URL=https://arb-mainnet.g.alchemy.com/v2/your-key

# Logging
RUST_LOG=info
```

---

## 📁 Project Structure

```
Nexuszero-Protocol/
├── Cargo.toml                      # Workspace configuration
├── services/
│   ├── api_gateway/
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── main.rs             # Entry point
│   │   │   ├── config.rs           # Configuration
│   │   │   ├── state.rs            # Shared state
│   │   │   ├── routes.rs           # Route definitions
│   │   │   ├── metrics.rs          # Prometheus metrics
│   │   │   ├── handlers/           # Request handlers
│   │   │   │   ├── mod.rs
│   │   │   │   ├── auth.rs
│   │   │   │   ├── transactions.rs
│   │   │   │   ├── privacy.rs
│   │   │   │   ├── compliance.rs
│   │   │   │   ├── bridge.rs
│   │   │   │   ├── health.rs
│   │   │   │   └── websocket.rs
│   │   │   └── middleware/         # Middleware stack
│   │   │       ├── mod.rs
│   │   │       ├── auth.rs
│   │   │       ├── rate_limit.rs
│   │   │       └── circuit_breaker.rs
│   │   └── migrations/
│   │
│   ├── transaction_service/
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── main.rs
│   │   │   ├── config.rs
│   │   │   ├── state.rs
│   │   │   ├── routes.rs
│   │   │   ├── mempool.rs          # Transaction mempool
│   │   │   ├── validator.rs        # Transaction validation
│   │   │   ├── processor.rs        # Transaction processing
│   │   │   └── handlers/
│   │   └── migrations/
│   │
│   ├── privacy_service/
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── main.rs
│   │   │   ├── config.rs
│   │   │   ├── state.rs
│   │   │   ├── routes.rs
│   │   │   ├── engine.rs           # APM privacy engine
│   │   │   ├── proof.rs            # ZKP generation
│   │   │   └── handlers/
│   │   └── migrations/
│   │
│   ├── compliance_service/
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── main.rs
│   │   │   ├── config.rs
│   │   │   ├── state.rs
│   │   │   ├── routes.rs
│   │   │   ├── rules.rs            # Rule engine
│   │   │   ├── screening.rs        # Sanctions screening
│   │   │   └── handlers/
│   │   └── migrations/
│   │
│   ├── bridge_service/
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── main.rs
│   │   │   ├── config.rs
│   │   │   ├── state.rs
│   │   │   ├── routes.rs
│   │   │   ├── htlc.rs             # HTLC implementation
│   │   │   ├── chain_client.rs     # Multi-chain client
│   │   │   └── handlers/
│   │   └── migrations/
│   │
│   ├── Dockerfile                  # Multi-stage build
│   └── docker-compose.services.yml
│
├── shared/
│   ├── nexuszero-common/
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── types.rs            # Common types
│   │       ├── error.rs            # Error types
│   │       ├── config.rs           # Config utilities
│   │       └── utils.rs            # Utility functions
│   │
│   ├── nexuszero-crypto-lib/
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── pedersen.rs         # Pedersen commitments
│   │       ├── zkp.rs              # ZKP primitives
│   │       ├── hash.rs             # Hashing utilities
│   │       └── signature.rs        # Signature schemes
│   │
│   └── nexuszero-db/
│       └── src/
│           ├── lib.rs
│           ├── postgres.rs         # PostgreSQL connection
│           └── redis.rs            # Redis connection
│
└── docs/
    └── PHASE1_IMPLEMENTATION.md    # This file
```

---

## 🔒 Security Considerations

### Implemented

- [x] JWT token-based authentication
- [x] Argon2id password hashing
- [x] Rate limiting (per-user and global)
- [x] Input validation on all endpoints
- [x] CORS protection
- [x] Circuit breaker for cascading failure prevention
- [x] Audit logging for compliance actions
- [x] Sanctions screening integration points

### Phase 2 Priorities

- [ ] Hardware security module (HSM) integration
- [ ] Multi-party computation (MPC) for key management
- [ ] Formal verification of ZKP circuits
- [ ] Security audit by third party

---

## 📈 Metrics & Monitoring

### Prometheus Metrics

Each service exposes metrics at `/metrics`:

```
# API Gateway
http_requests_total{method, path, status}
http_request_duration_seconds{method, path}
rate_limit_hits_total{user_id}
circuit_breaker_state{service}
active_websocket_connections

# Transaction Service
transactions_submitted_total{type}
transactions_processed_total{status}
mempool_size
transaction_validation_duration_seconds

# Privacy Service
proofs_generated_total{type, level}
proof_generation_duration_seconds{type}
proofs_verified_total{result}

# Compliance Service
compliance_checks_total{result, jurisdiction}
sanctions_screenings_total{result}

# Bridge Service
bridge_transfers_total{source, destination, status}
htlc_created_total{chain}
htlc_claimed_total{chain}
liquidity_pool_balance{chain, asset}
```

### Grafana Dashboards

Pre-configured dashboards available in `grafana/dashboards/`:

- Service Overview
- Transaction Pipeline
- Privacy Metrics
- Compliance Monitoring
- Bridge Activity

---

## 🧪 Testing

```bash
# Run all tests
cargo test --workspace

# Run tests with coverage
cargo tarpaulin --workspace --out html

# Run specific service tests
cargo test --package api_gateway
cargo test --package privacy_service

# Run integration tests
cargo test --workspace -- --ignored
```

---

## 📝 API Documentation

### Authentication

```bash
# Register
curl -X POST http://localhost:8080/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "SecurePass123!", "username": "user1"}'

# Login
curl -X POST http://localhost:8080/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "SecurePass123!"}'
```

### Transactions

```bash
# Create transaction
curl -X POST http://localhost:8080/v1/transactions \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "from": "sender_address",
    "to": "recipient_address",
    "amount": "100.00",
    "asset": "NZT",
    "privacy_level": "enhanced"
  }'

# Get transaction status
curl http://localhost:8080/v1/transactions/<tx_id> \
  -H "Authorization: Bearer <token>"
```

### Bridge

```bash
# Get quote
curl -X POST http://localhost:8080/v1/bridge/quote \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "source_chain": "ethereum",
    "destination_chain": "polygon",
    "asset": "USDC",
    "amount": "1000.00"
  }'

# Initiate transfer
curl -X POST http://localhost:8080/v1/bridge/transfer \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "quote_id": "<quote_id>",
    "source_address": "0x...",
    "destination_address": "0x..."
  }'
```

---

## 🔄 Phase 2 Roadmap

1. **Ring Signatures** - Enhanced privacy with ring signature implementation
2. **Multi-Chain Expansion** - Solana, Cosmos, Polkadot support
3. **MPC Key Management** - Threshold signatures for custody
4. **Governance Module** - On-chain governance for protocol upgrades
5. **Staking System** - Validator staking and rewards
6. **OpenTelemetry** - Distributed tracing across services
7. **GraphQL API** - Alternative API interface
8. **Mobile SDK** - iOS and Android integration

---

## 📄 License

MIT License - See [LICENSE](../LICENSE) for details.

---

## 🤝 Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

---

**Built with ❤️ by the NexusZero Protocol Team**
