# Nexuszero Protocol - Deployment Guide

This guide covers deploying the Nexuszero Protocol stack using Docker Compose (for local development) and Kubernetes (for production).

## Table of Contents

- [Prerequisites](#prerequisites)
- [Local Development with Docker Compose](#local-development-with-docker-compose)
- [Production Deployment with Kubernetes](#production-deployment-with-kubernetes)
- [CI/CD Pipeline](#cicd-pipeline)
- [Monitoring and Observability](#monitoring-and-observability)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### For Docker Compose (Local Development)

- Docker Desktop or Docker Engine 20.10+
- Docker Compose V2
- At least 8GB RAM available for Docker
- At least 20GB disk space

### For Kubernetes (Production)

- Kubernetes cluster (v1.24+)
  - GKE, EKS, AKS, or self-hosted
- kubectl CLI installed and configured
- Helm 3.x (optional, for easier deployments)
- Ingress controller (nginx-ingress recommended)
- cert-manager (for TLS certificates)
- Persistent storage provisioner

### For CI/CD

- GitHub repository with Actions enabled
- GitHub Container Registry access (or alternative container registry)
- Kubernetes cluster credentials for deployment

## Local Development with Docker Compose

### Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/iamthegreatdestroyer/Nexuszero-Protocol.git
   cd Nexuszero-Protocol
   ```

2. **Start all services:**
   ```bash
   docker-compose up -d
   ```

3. **Check service status:**
   ```bash
   docker-compose ps
   ```

4. **View logs:**
   ```bash
   docker-compose logs -f
   ```

### Services and Ports

| Service | Port | URL |
|---------|------|-----|
| Nexuszero Crypto | 13001 | http://localhost:13001 |
| Nexuszero Optimizer | 13002 | http://localhost:13002 |
| Nexuszero Monitor | 13003 | http://localhost:13003 |
| Prometheus | 9090 | http://localhost:9090 |
| Grafana | 3000 | http://localhost:3000 |
| Loki | 3100 | http://localhost:3100 |

### Accessing Services

**Grafana Dashboard:**
- URL: http://localhost:3000
- Username: `admin`
- Password: `admin` (change on first login)

**Prometheus:**
- URL: http://localhost:9090
- Check targets at: http://localhost:9090/targets

### Building Images Locally

Build specific services:

```bash
# Build Rust services
docker-compose build nexuszero-crypto

# Build Python optimizer
docker-compose build nexuszero-optimizer

# Build all services
docker-compose build
```

### Development Workflow

1. Make code changes in your editor
2. Rebuild the affected service:
   ```bash
   docker-compose build nexuszero-optimizer
   ```
3. Restart the service:
   ```bash
   docker-compose up -d nexuszero-optimizer
   ```
4. Check logs:
   ```bash
   docker-compose logs -f nexuszero-optimizer
   ```

### Stopping Services

```bash
# Stop all services
docker-compose stop

# Stop and remove containers
docker-compose down

# Stop and remove containers, volumes, and images
docker-compose down -v --rmi all
```

## Production Deployment with Kubernetes

### Step 1: Prepare Your Cluster

1. **Create a namespace:**
   ```bash
   kubectl apply -f k8s/namespace.yaml
   ```

2. **Configure secrets:**
   
   Edit `k8s/secrets.yaml` with your actual secrets (DO NOT commit to Git):
   
   ```bash
   # Create secrets from environment file
   kubectl create secret generic nexuszero-secrets \
     --from-env-file=.env.production \
     --namespace=nexuszero
   
   # Or create secrets individually
   kubectl create secret generic nexuszero-secrets \
     --from-literal=GRAFANA_ADMIN_PASSWORD='your-secure-password' \
     --namespace=nexuszero
   ```

3. **Apply ConfigMaps:**
   ```bash
   kubectl apply -f k8s/configmap.yaml
   ```

### Step 2: Setup RBAC and Storage

```bash
# Create service accounts and permissions
kubectl apply -f k8s/rbac.yaml

# Create persistent volume claims
kubectl apply -f k8s/pvc.yaml

# Verify PVCs are bound
kubectl get pvc -n nexuszero
```

### Step 3: Deploy Applications

```bash
# Deploy all services
kubectl apply -f k8s/deployment.yaml

# Create services
kubectl apply -f k8s/service.yaml

# Check deployment status
kubectl get deployments -n nexuszero
kubectl get pods -n nexuszero

# Wait for all pods to be ready
kubectl wait --for=condition=ready pod -l tier=core -n nexuszero --timeout=300s
```

### Step 4: Configure Ingress

1. **Edit ingress.yaml** with your domain names
2. **Apply ingress:**
   ```bash
   kubectl apply -f k8s/ingress.yaml
   ```
3. **Check ingress:**
   ```bash
   kubectl get ingress -n nexuszero
   ```

### Step 5: Verify Deployment

```bash
# Check all resources
kubectl get all -n nexuszero

# Check pod logs
kubectl logs -n nexuszero -l app=nexuszero-crypto

# Test service connectivity
kubectl port-forward -n nexuszero svc/grafana 3000:3000
```

### Scaling

Scale deployments as needed:

```bash
# Scale crypto service
kubectl scale deployment/nexuszero-crypto -n nexuszero --replicas=3

# Auto-scaling with HPA
kubectl autoscale deployment nexuszero-crypto \
  --cpu-percent=70 \
  --min=2 \
  --max=10 \
  -n nexuszero
```

### Updating Deployments

```bash
# Update image version
kubectl set image deployment/nexuszero-crypto \
  nexuszero-crypto=ghcr.io/iamthegreatdestroyer/nexuszero-protocol/nexuszero-crypto:v1.2.0 \
  -n nexuszero

# Rollout status
kubectl rollout status deployment/nexuszero-crypto -n nexuszero

# Rollback if needed
kubectl rollout undo deployment/nexuszero-crypto -n nexuszero
```

## CI/CD Pipeline

### GitHub Actions Workflow

The CI/CD pipeline (`.github/workflows/deploy.yml`) automatically:

1. **Runs tests** on every push and PR
2. **Builds Docker images** on push to main
3. **Pushes images** to GitHub Container Registry
4. **Deploys to staging** automatically on main branch
5. **Deploys to production** with manual approval

### Setting Up CI/CD

1. **Add Kubernetes credentials as GitHub Secrets:**
   
   Go to your repository → Settings → Secrets and variables → Actions:
   
   - `KUBE_CONFIG_STAGING`: Your staging cluster kubeconfig
   - `KUBE_CONFIG_PRODUCTION`: Your production cluster kubeconfig

2. **Configure environments:**
   
   Go to Settings → Environments:
   
   - Create "staging" environment (auto-deploy)
   - Create "production" environment (with required reviewers)

3. **First deployment:**
   ```bash
   # Trigger manual deployment
   gh workflow run deploy.yml -f environment=staging
   ```

### Manual Deployment

If you need to deploy manually:

```bash
# Build images
docker build -t ghcr.io/iamthegreatdestroyer/nexuszero-protocol/nexuszero-crypto:v1.0.0 .
docker build -t ghcr.io/iamthegreatdestroyer/nexuszero-protocol/nexuszero-optimizer:v1.0.0 ./nexuszero-optimizer

# Push images
docker push ghcr.io/iamthegreatdestroyer/nexuszero-protocol/nexuszero-crypto:v1.0.0
docker push ghcr.io/iamthegreatdestroyer/nexuszero-protocol/nexuszero-optimizer:v1.0.0

# Update Kubernetes deployment
kubectl set image deployment/nexuszero-crypto \
  nexuszero-crypto=ghcr.io/iamthegreatdestroyer/nexuszero-protocol/nexuszero-crypto:v1.0.0 \
  -n nexuszero
```

## Monitoring and Observability

### Accessing Grafana

**Via Port Forward:**
```bash
kubectl port-forward -n nexuszero svc/grafana 3000:3000
```
Visit: http://localhost:3000

**Via LoadBalancer (if configured):**
```bash
kubectl get svc grafana -n nexuszero
```
Use the EXTERNAL-IP shown.

**Via Ingress (if configured):**
Visit: https://grafana.nexuszero.io

### Default Dashboards

1. **Nexuszero Overview** - Main system metrics
   - Request rates
   - Latency percentiles
   - Error rates
   - Service logs

2. **Custom Dashboards** - Import from `grafana/dashboards/`

### Prometheus Queries

Access Prometheus:
```bash
kubectl port-forward -n nexuszero svc/prometheus 9090:9090
```

Common queries:
- Request rate: `rate(nexuszero_request_count_total[5m])`
- Error rate: `rate(nexuszero_error_count_total[5m])`
- P99 latency: `histogram_quantile(0.99, rate(nexuszero_request_latency_bucket[5m]))`

### Loki Logs

Query logs in Grafana using LogQL:

```logql
# All logs from nexuszero services
{service=~"nexuszero.*"}

# Errors only
{service=~"nexuszero.*"} |= "error"

# Specific service
{service="nexuszero-crypto"}
```

## Troubleshooting

### Docker Compose Issues

**Services won't start:**
```bash
# Check logs
docker-compose logs

# Check if ports are in use
netstat -an | grep 13001

# Rebuild images
docker-compose build --no-cache
docker-compose up -d
```

**Out of memory:**
```bash
# Increase Docker memory in Docker Desktop settings
# Or clean up
docker system prune -a
```

### Kubernetes Issues

**Pods not starting:**
```bash
# Describe pod
kubectl describe pod -n nexuszero <pod-name>

# Check events
kubectl get events -n nexuszero --sort-by='.lastTimestamp'

# Check logs
kubectl logs -n nexuszero <pod-name>
```

**Image pull errors:**
```bash
# Create image pull secret
kubectl create secret docker-registry ghcr-secret \
  --docker-server=ghcr.io \
  --docker-username=$GITHUB_USERNAME \
  --docker-password=$GITHUB_TOKEN \
  -n nexuszero

# Add to deployment
spec:
  imagePullSecrets:
  - name: ghcr-secret
```

**PVC stuck in Pending:**
```bash
# Check storage classes
kubectl get sc

# Check PVC details
kubectl describe pvc -n nexuszero <pvc-name>

# May need to provision storage or change storageClassName
```

**Services not accessible:**
```bash
# Check service endpoints
kubectl get endpoints -n nexuszero

# Check network policies
kubectl get networkpolicies -n nexuszero

# Test from inside cluster
kubectl run -it --rm debug --image=alpine --restart=Never -- sh
apk add curl
curl http://nexuszero-crypto.nexuszero.svc.cluster.local:13001
```

### CI/CD Issues

**Build failures:**
- Check GitHub Actions logs
- Verify Dockerfile syntax
- Test build locally: `docker build .`

**Deployment failures:**
- Check Kubernetes credentials
- Verify cluster access: `kubectl get nodes`
- Check resource quotas: `kubectl describe resourcequota -n nexuszero`

**Registry push failures:**
- Verify GitHub token permissions
- Check registry quotas
- Verify image tags

## Security Best Practices

1. **Secrets Management:**
   - Never commit secrets to Git
   - Use external secrets manager (Vault, AWS Secrets Manager)
   - Rotate secrets regularly

2. **Network Security:**
   - Implement NetworkPolicies
   - Use TLS for all external traffic
   - Restrict ingress/egress

3. **RBAC:**
   - Follow principle of least privilege
   - Regular audit of permissions
   - Use service accounts appropriately

4. **Container Security:**
   - Scan images for vulnerabilities (Trivy, Snyk)
   - Use non-root users in containers
   - Keep base images updated

5. **Monitoring:**
   - Set up alerts for security events
   - Monitor authentication failures
   - Track unusual resource usage

## Backup and Recovery

### Database Backups (if applicable)

```bash
# Backup PVC data
kubectl exec -n nexuszero <pod-name> -- tar czf /tmp/backup.tar.gz /data
kubectl cp nexuszero/<pod-name>:/tmp/backup.tar.gz ./backup.tar.gz
```

### Configuration Backups

```bash
# Export all resources
kubectl get all -n nexuszero -o yaml > nexuszero-backup.yaml

# Backup secrets (encrypted)
kubectl get secrets -n nexuszero -o yaml > secrets-backup.yaml
```

### Disaster Recovery

1. Maintain infrastructure as code (already done with k8s manifests)
2. Regular backups of persistent data
3. Document recovery procedures
4. Test recovery process regularly

## Performance Tuning

### Resource Requests/Limits

Adjust in `k8s/deployment.yaml`:

```yaml
resources:
  requests:
    memory: "256Mi"
    cpu: "250m"
  limits:
    memory: "512Mi"
    cpu: "500m"
```

### Database/Storage Optimization

- Use appropriate storage classes (SSD for performance)
- Tune PVC sizes based on actual usage
- Implement data retention policies

### Application Tuning

- Adjust worker threads/processes
- Configure connection pools
- Enable caching where appropriate

## Support and Documentation

- **General Documentation:** [README.md](../README.md)
- **Metrics Documentation:** [METRICS_ENDPOINTS.md](./METRICS_ENDPOINTS.md)
- **Kubernetes Deployment:** [k8s/README.md](../k8s/README.md)
- **Project Overview:** [PROJECT_OVERVIEW.md](../PROJECT_OVERVIEW.md)

## Additional Resources

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
