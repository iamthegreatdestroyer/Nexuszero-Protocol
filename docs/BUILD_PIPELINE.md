# 🔧 Build & Deployment Pipeline Documentation

## Overview

The NexusZero Protocol uses a comprehensive CI/CD pipeline designed for:

- **Reproducible builds** - Same input always produces same output
- **Security scanning** - Continuous vulnerability detection
- **Multi-platform support** - Linux, Windows, macOS
- **Fast feedback** - Parallel jobs with intelligent caching

---

## 📁 Pipeline Files

| Workflow                  | Purpose                             | Trigger                  |
| ------------------------- | ----------------------------------- | ------------------------ |
| `rust.yml`                | Rust CI - format, lint, test, build | Push/PR to main/develop  |
| `security-scan.yml`       | Dependency vulnerability scanning   | Push/PR + daily schedule |
| `reproducible-builds.yml` | Deterministic build verification    | Push/PR + tags           |
| `deploy.yml`              | Build & deploy Docker images        | Push to main + manual    |

---

## 🔐 Security Scanning

### Rust Dependencies

- **cargo-audit**: Check for known CVEs in Cargo dependencies
- **cargo-deny**: Policy enforcement for licenses, sources, advisories

### TypeScript/Node.js

- **npm audit**: Check for vulnerabilities in npm packages
- **license-checker**: Ensure license compliance

### Python

- **safety**: Scan requirements.txt against vulnerability database
- **pip-audit**: Modern pip vulnerability scanner

### Containers

- **Trivy**: Scan Docker images and filesystem for vulnerabilities
- **Gitleaks/TruffleHog**: Secret detection in code

### Software Bill of Materials (SBOM)

- **Syft**: Generate SPDX and CycloneDX format SBOMs
- Useful for compliance and supply chain security

---

## 🔄 Reproducible Builds

### Key Principles

1. **Lockfile Enforcement**

   - `Cargo.lock` must exist and be committed
   - `package-lock.json` required for all npm packages
   - `--locked` flag used in all cargo commands

2. **Deterministic Build Settings**

   ```bash
   # Environment variables for reproducibility
   SOURCE_DATE_EPOCH=0
   CARGO_INCREMENTAL=0
   RUSTFLAGS="--remap-path-prefix=/build=/app"
   ```

3. **Pinned Versions**

   - Rust version: 1.75.0 (exact)
   - Node.js version: 20.10.0 (exact)
   - Base images: `debian:bookworm-20241111-slim` (dated tags)

4. **Build Verification**
   - Build twice, compare hashes
   - Upload provenance for releases

### Provenance Generation

For tagged releases, the pipeline generates:

- Build provenance (SLSA-style)
- SHA256/SHA512 checksums
- Cosign signatures (keyless via Sigstore)

---

## 🐳 Docker Best Practices

### Multi-Stage Builds

```dockerfile
# Stage 1: Builder (with build tools)
FROM rust:1.75.0-slim-bookworm as builder
# ... build steps ...

# Stage 2: Runtime (minimal, secure)
FROM debian:bookworm-20241111-slim as runtime
# ... only runtime deps ...
```

### Security Hardening

- Non-root user: `nexuszero`
- Minimal base image
- No shell in production (optional)
- Read-only filesystem (in K8s)

### Dependency Caching

```dockerfile
# Copy lockfiles first
COPY Cargo.lock Cargo.toml ./
# Create dummy sources
RUN mkdir src && echo "fn main(){}" > src/lib.rs
# Build deps (cached)
RUN cargo build --release
# Copy real sources
COPY src ./src
# Build app (fast, deps cached)
RUN cargo build --release
```

---

## 📋 Configuration Files

### `deny.toml` - Cargo Deny Policy

```toml
[advisories]
vulnerability = "deny"
unmaintained = "warn"

[licenses]
allow = ["MIT", "Apache-2.0", "BSD-3-Clause"]
unlicensed = "deny"

[bans]
multiple-versions = "warn"
wildcards = "deny"
```

### `.npmrc` - NPM Configuration

```ini
save-exact=true
package-lock=true
audit=true
```

---

## 🚀 Running Locally

### Security Scan

```bash
# Rust
cargo install cargo-audit cargo-deny
cargo audit
cargo deny check

# NPM
cd nexuszero-sdk && npm audit

# Python
pip install safety pip-audit
safety check -r requirements.txt
```

### Reproducible Build

```bash
# Generate lockfile
cargo generate-lockfile

# Build with locked deps
cargo build --release --locked

# Verify hash
sha256sum target/release/nexuszero-*
```

### Docker Build

```bash
# Reproducible build
docker build \
  --build-arg SOURCE_DATE_EPOCH=0 \
  --build-arg RUST_VERSION=1.75.0 \
  -t nexuszero-crypto:local .

# Scan image
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image nexuszero-crypto:local
```

---

## 📊 Metrics & Monitoring

### Build Metrics

- Build duration per crate
- Cache hit rate
- Test pass rate

### Security Metrics

- Vulnerabilities by severity (critical/high/medium/low)
- Time to remediate
- License compliance percentage

### Artifacts

All workflows upload artifacts for debugging:

- Audit results (JSON + logs)
- Build hashes
- SBOM files
- Provenance

---

## 🔧 Troubleshooting

### Cargo.lock Mismatch

```bash
# Regenerate lockfile
cargo generate-lockfile
git add Cargo.lock
git commit -m "chore: update Cargo.lock"
```

### Vulnerability Found

1. Check advisory details in cargo-audit output
2. Update the affected dependency: `cargo update -p <crate>`
3. If no fix available, add to ignore list in `deny.toml` with justification

### Build Not Reproducible

- Check for timestamps in build artifacts
- Ensure `SOURCE_DATE_EPOCH` is set
- Verify `--remap-path-prefix` is used
- Check for non-deterministic proc macros

---

## 📚 References

- [Reproducible Builds](https://reproducible-builds.org/)
- [SLSA Framework](https://slsa.dev/)
- [cargo-deny](https://embarkstudios.github.io/cargo-deny/)
- [Trivy](https://aquasecurity.github.io/trivy/)
- [Sigstore/Cosign](https://docs.sigstore.dev/)
