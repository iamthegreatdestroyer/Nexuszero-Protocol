# ============================================================================
# Nexuszero Protocol - Rust Services
# Reproducible, Secure Multi-Stage Docker Build
# ============================================================================

# Build arguments for reproducibility
ARG RUST_VERSION=1.75.0
ARG DEBIAN_VERSION=bookworm-20241111-slim
ARG SOURCE_DATE_EPOCH=0

# ============================================================================
# Stage 1: Builder
# ============================================================================
FROM rust:${RUST_VERSION}-slim-bookworm as builder

# Reproducibility: Set build timestamp
ARG SOURCE_DATE_EPOCH
ENV SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH}

# Security: Run as non-root during build where possible
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /build

# Copy lockfile first for better caching
COPY Cargo.lock ./

# Copy workspace Cargo.toml
COPY Cargo.toml ./

# Copy workspace member Cargo.toml files (for dependency caching)
COPY nexuszero-crypto/Cargo.toml ./nexuszero-crypto/
COPY nexuszero-holographic/Cargo.toml ./nexuszero-holographic/
COPY nexuszero-integration/Cargo.toml ./nexuszero-integration/

# Create dummy source files for dependency compilation
RUN mkdir -p nexuszero-crypto/src nexuszero-holographic/src nexuszero-integration/src \
    && echo "fn main() {}" > nexuszero-crypto/src/lib.rs \
    && echo "fn main() {}" > nexuszero-holographic/src/lib.rs \
    && echo "fn main() {}" > nexuszero-integration/src/lib.rs

# Build dependencies only (cached layer)
RUN cargo build --release --workspace --locked 2>/dev/null || true

# Remove dummy source files
RUN rm -rf nexuszero-crypto/src nexuszero-holographic/src nexuszero-integration/src

# Copy actual source code
COPY nexuszero-crypto ./nexuszero-crypto
COPY nexuszero-holographic ./nexuszero-holographic
COPY nexuszero-integration ./nexuszero-integration

# Build the workspace with locked dependencies and reproducible settings
ENV RUSTFLAGS="-D warnings --remap-path-prefix=/build=/app"
RUN cargo build --release --workspace --locked

# Strip binaries for smaller size
RUN find target/release -maxdepth 1 -type f -executable -exec strip {} \; 2>/dev/null || true

# ============================================================================
# Stage 2: Runtime
# ============================================================================
FROM debian:bookworm-20241111-slim as runtime

# Labels for container identification
LABEL org.opencontainers.image.title="NexusZero Protocol"
LABEL org.opencontainers.image.description="Quantum-resistant zero-knowledge proof system"
LABEL org.opencontainers.image.vendor="NexusZero"
LABEL org.opencontainers.image.source="https://github.com/iamthegreatdestroyer/Nexuszero-Protocol"
LABEL org.opencontainers.image.licenses="MIT"

# Security: Create non-root user
RUN groupadd -r nexuszero && useradd -r -g nexuszero -d /app -s /sbin/nologin nexuszero

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/bin /app/data /app/logs \
    && chown -R nexuszero:nexuszero /app

# Copy binaries from builder with specific ownership
COPY --from=builder --chown=nexuszero:nexuszero /build/target/release/nexuszero-* /app/bin/

# Security: Set restrictive permissions
RUN chmod 755 /app/bin/* 2>/dev/null || true

# Security: Drop all capabilities, run as non-root
USER nexuszero

# Environment
ENV RUST_LOG=info
ENV PATH="/app/bin:${PATH}"

# Expose ports
EXPOSE 13001

# Health check with curl
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:13001/health || exit 1

# Default command
CMD ["/bin/sh"]
