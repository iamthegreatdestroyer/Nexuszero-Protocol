# Kubernetes Deployment Guide

This directory contains Kubernetes manifests for deploying the Nexuszero Protocol stack.

## Prerequisites

- Kubernetes cluster (v1.24+)
- kubectl configured to access your cluster
- Ingress controller (e.g., nginx-ingress)
- cert-manager (for TLS certificates, optional)
- Storage class configured for PersistentVolumes

## Quick Start

### 1. Create Namespace and ConfigMaps

```bash
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f secrets.yaml
```

### 2. Create RBAC Resources

```bash
kubectl apply -f rbac.yaml
```

### 3. Create Persistent Volume Claims

```bash
kubectl apply -f pvc.yaml
```

### 4. Deploy Services

```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

### 5. Configure Ingress (Optional)

Edit `ingress.yaml` to match your domain names, then:

```bash
kubectl apply -f ingress.yaml
```

## Deployment Order

For a complete deployment, apply manifests in this order:

```bash
# Step 1: Namespace and configuration
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml

# Step 2: Secrets (update with real values first!)
kubectl apply -f secrets.yaml

# Step 3: RBAC
kubectl apply -f rbac.yaml

# Step 4: Storage
kubectl apply -f pvc.yaml

# Step 5: Applications
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

# Step 6: Ingress (optional)
kubectl apply -f ingress.yaml
```

Or apply all at once:

```bash
kubectl apply -f .
```

## Verify Deployment

Check all resources are running:

```bash
kubectl get all -n nexuszero
```

Check pod status:

```bash
kubectl get pods -n nexuszero
```

Check services:

```bash
kubectl get svc -n nexuszero
```

View logs:

```bash
# View logs for a specific pod
kubectl logs -n nexuszero <pod-name>

# Follow logs
kubectl logs -n nexuszero <pod-name> -f

# View logs for all pods of a service
kubectl logs -n nexuszero -l app=nexuszero-crypto
```

## Access Services

### Port Forwarding (for testing)

```bash
# Access Grafana
kubectl port-forward -n nexuszero svc/grafana 3000:3000

# Access Prometheus
kubectl port-forward -n nexuszero svc/prometheus 9090:9090

# Access Nexuszero Crypto
kubectl port-forward -n nexuszero svc/nexuszero-crypto 13001:13001
```

Then visit:
- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090
- Nexuszero Crypto: http://localhost:13001

### LoadBalancer (Cloud environments)

If using LoadBalancer services:

```bash
kubectl get svc -n nexuszero grafana
```

Look for the EXTERNAL-IP and access via that IP on port 3000.

### Ingress (Production)

Access via your configured domain names:
- https://api.nexuszero.io
- https://grafana.nexuszero.io
- etc.

## Scaling

Scale deployments:

```bash
kubectl scale deployment/nexuszero-crypto -n nexuszero --replicas=3
kubectl scale deployment/nexuszero-optimizer -n nexuszero --replicas=2
```

## Updates

Update image versions:

```bash
kubectl set image deployment/nexuszero-crypto -n nexuszero \
  nexuszero-crypto=ghcr.io/iamthegreatdestroyer/nexuszero-crypto:v1.2.0
```

Or edit the deployment:

```bash
kubectl edit deployment/nexuszero-crypto -n nexuszero
```

## Monitoring

### Check Prometheus Targets

```bash
kubectl port-forward -n nexuszero svc/prometheus 9090:9090
```

Visit http://localhost:9090/targets

### Access Grafana Dashboards

```bash
kubectl port-forward -n nexuszero svc/grafana 3000:3000
```

Visit http://localhost:3000 (default: admin/admin)

## Troubleshooting

### Check Pod Status

```bash
kubectl describe pod -n nexuszero <pod-name>
```

### View Events

```bash
kubectl get events -n nexuszero --sort-by='.lastTimestamp'
```

### Execute Commands in Pod

```bash
kubectl exec -it -n nexuszero <pod-name> -- /bin/sh
```

### Check Resource Usage

```bash
kubectl top pods -n nexuszero
kubectl top nodes
```

### Delete and Recreate

```bash
# Delete all resources
kubectl delete -f .

# Recreate
kubectl apply -f .
```

## Storage Classes

The PVCs use `storageClassName: standard`. Adjust this based on your cluster:

- **GKE**: `standard`, `standard-rwo`, `premium-rwo`
- **EKS**: `gp2`, `gp3`, `io1`
- **AKS**: `default`, `managed-premium`
- **On-premise**: Check with your cluster admin

## Security Considerations

1. **Secrets**: Update `secrets.yaml` with actual secrets before deploying
2. **Network Policies**: Consider adding NetworkPolicy resources
3. **Pod Security**: Add SecurityContext to pods
4. **RBAC**: Review and adjust service account permissions
5. **TLS**: Configure cert-manager for automatic TLS certificates

## Cleanup

Remove all resources:

```bash
kubectl delete namespace nexuszero
```

Or delete individual resources:

```bash
kubectl delete -f ingress.yaml
kubectl delete -f service.yaml
kubectl delete -f deployment.yaml
kubectl delete -f pvc.yaml
kubectl delete -f rbac.yaml
kubectl delete -f secrets.yaml
kubectl delete -f configmap.yaml
kubectl delete -f namespace.yaml
```
