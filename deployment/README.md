# Nexuszero Protocol Deployment

## Quick Start

```bash
# Deploy all services
./deploy.sh

# Check status
docker-compose -f docker-compose.edge.yml ps

# View logs
docker-compose -f docker-compose.edge.yml logs -f nexuszero-api
```

## Services

| Service       | Port       | Description    |
| ------------- | ---------- | -------------- |
| nexuszero-api | 8080       | Main API       |
| neo4j         | 7474, 7687 | Graph Database |
| redis         | 6379       | Cache          |
| prometheus    | 9090       | Metrics        |

## API Endpoints

- `GET /health` - Health check
- `POST /api/v1/optimization/submit` - Submit optimization job
- `GET /api/v1/optimization/status/{job_id}` - Get job status
- `POST /api/v1/graph/query` - Query primitives
- `GET /api/v1/graph/primitives` - List primitives
- `POST /auth/wallet` - Wallet authentication
- `POST /auth/refresh` - Refresh tokens

## Environment Variables

| Variable              | Default                      | Description        |
| --------------------- | ---------------------------- | ------------------ |
| NEO4J_URI             | bolt://neo4j:7687            | Neo4j connection   |
| NEO4J_USER            | neo4j                        | Neo4j username     |
| NEO4J_PASSWORD        | nexuszero                    | Neo4j password     |
| JWT_PRIVATE_KEY_PATH  | /app/secrets/jwt_private.pem | JWT key path       |
| DATA_SOVEREIGNTY_MODE | strict                       | Data handling mode |
