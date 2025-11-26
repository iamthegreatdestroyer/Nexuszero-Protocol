# ============================================================================
# NexusZero Protocol - Production Multi-Stage Dockerfile
# ============================================================================
# Build Stage: Compiles all Rust services
# Runtime Stage: Minimal runtime with compiled binaries
# ============================================================================

# Build stage
FROM rust:1.82-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    libpq-dev \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy workspace files
COPY Cargo.toml Cargo.lock ./
COPY nexuszero-crypto ./nexuszero-crypto
COPY nexuszero-holographic ./nexuszero-holographic
COPY nexuszero-integration ./nexuszero-integration
COPY shared ./shared
COPY services ./services

# Build all services in release mode
RUN cargo build --release --workspace

# ============================================================================
# Runtime stage - API Gateway
# ============================================================================
FROM debian:bookworm-slim AS api-gateway

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/target/release/api_gateway /app/api_gateway

# Create non-root user
RUN useradd -ms /bin/bash nexuszero
USER nexuszero

ENV RUST_LOG=info
EXPOSE 8080

CMD ["./api_gateway"]

# ============================================================================
# Runtime stage - Transaction Service
# ============================================================================
FROM debian:bookworm-slim AS transaction-service

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/target/release/transaction_service /app/transaction_service

RUN useradd -ms /bin/bash nexuszero
USER nexuszero

ENV RUST_LOG=info
EXPOSE 8081

CMD ["./transaction_service"]

# ============================================================================
# Runtime stage - Privacy Service
# ============================================================================
FROM debian:bookworm-slim AS privacy-service

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/target/release/privacy_service /app/privacy_service

RUN useradd -ms /bin/bash nexuszero
USER nexuszero

ENV RUST_LOG=info
EXPOSE 8082

CMD ["./privacy_service"]

# ============================================================================
# Runtime stage - Compliance Service
# ============================================================================
FROM debian:bookworm-slim AS compliance-service

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/target/release/compliance_service /app/compliance_service

RUN useradd -ms /bin/bash nexuszero
USER nexuszero

ENV RUST_LOG=info
EXPOSE 8083

CMD ["./compliance_service"]

# ============================================================================
# Runtime stage - Bridge Service
# ============================================================================
FROM debian:bookworm-slim AS bridge-service

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/target/release/bridge_service /app/bridge_service

RUN useradd -ms /bin/bash nexuszero
USER nexuszero

ENV RUST_LOG=info
EXPOSE 8084

CMD ["./bridge_service"]
