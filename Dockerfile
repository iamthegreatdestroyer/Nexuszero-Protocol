# Nexuszero Protocol - Rust Services
FROM rust:1.75-slim as builder

# Install dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy workspace files
COPY Cargo.toml Cargo.lock ./
COPY nexuszero-crypto ./nexuszero-crypto
COPY nexuszero-holographic ./nexuszero-holographic
COPY nexuszero-integration ./nexuszero-integration

# Build the workspace
RUN cargo build --release --workspace

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy binaries from builder
COPY --from=builder /app/target/release/nexuszero-* /app/bin/ 2>/dev/null || true

# Expose ports (can be overridden)
EXPOSE 13001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD true

# Default command - can be overridden
CMD ["/bin/sh"]
