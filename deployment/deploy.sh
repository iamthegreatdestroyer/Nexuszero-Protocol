#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Nexuszero Protocol Deployment ==="

# Check prerequisites
command -v docker >/dev/null 2>&1 || { echo "Docker required.  Aborting."; exit 1; }
command -v docker-compose >/dev/null 2>&1 || { echo "Docker Compose required. Aborting."; exit 1; }

# Create secrets directory
mkdir -p "$SCRIPT_DIR/secrets"

# Generate JWT keys if not present
if [ ! -f "$SCRIPT_DIR/secrets/jwt_private.pem" ]; then
    echo "[1/4] Generating JWT keys..."
    openssl genrsa -out "$SCRIPT_DIR/secrets/jwt_private.pem" 2048
    openssl rsa -in "$SCRIPT_DIR/secrets/jwt_private.pem" -pubout -out "$SCRIPT_DIR/secrets/jwt_public.pem"
else
    echo "[1/4] JWT keys exist, skipping..."
fi

# Build images
echo "[2/4] Building Docker images..."
cd "$SCRIPT_DIR"
docker-compose -f docker-compose.edge.yml build

# Stop existing
echo "[3/4] Stopping existing containers..."
docker-compose -f docker-compose.edge.yml down --remove-orphans || true

# Start services
echo "[4/4] Starting services..."
docker-compose -f docker-compose.edge.yml up -d

echo ""
echo "=== Deployment Complete ==="
echo "API: http://localhost:8080"
echo "Neo4j Browser: http://localhost:7474"
echo "Prometheus: http://localhost:9090"
echo ""
echo "Check status: docker-compose -f docker-compose.edge.yml ps"
echo "View logs: docker-compose -f docker-compose.edge.yml logs -f"
