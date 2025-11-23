# Prometheus Metrics Endpoints

This document describes the metrics exposed by Nexuszero Protocol services.

## Overview

All Nexuszero services expose Prometheus-compatible metrics on the `/metrics` endpoint. These metrics are automatically scraped by Prometheus and visualized in Grafana.

## Service Endpoints

| Service | Endpoint | Port |
|---------|----------|------|
| nexuszero-crypto | http://nexuszero-crypto:13001/metrics | 13001 |
| nexuszero-optimizer | http://nexuszero-optimizer:13002/metrics | 13002 |
| nexuszero-monitor | http://nexuszero-monitor:13003/metrics | 13003 |

## Standard Metrics

All services expose these standard metrics:

### Request Metrics

#### `nexuszero_request_count_total`
- **Type:** Counter
- **Description:** Total number of requests processed
- **Labels:**
  - `service`: Service name (e.g., "nexuszero-crypto")
  - `method`: HTTP method (GET, POST, etc.)
  - `endpoint`: Request endpoint path
  - `status_code`: HTTP response status code

**Example:**
```promql
# Rate of requests per second
rate(nexuszero_request_count_total[5m])

# Requests by service
sum(rate(nexuszero_request_count_total[5m])) by (service)

# 4xx/5xx error rate
sum(rate(nexuszero_request_count_total{status_code=~"4..|5.."}[5m])) by (service)
```

#### `nexuszero_request_latency_seconds`
- **Type:** Histogram
- **Description:** Request latency in seconds
- **Labels:**
  - `service`: Service name
  - `method`: HTTP method
  - `endpoint`: Request endpoint path
- **Buckets:** 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0

**Example:**
```promql
# P95 latency
histogram_quantile(0.95, rate(nexuszero_request_latency_seconds_bucket[5m]))

# P99 latency by service
histogram_quantile(0.99, sum(rate(nexuszero_request_latency_seconds_bucket[5m])) by (service, le))

# Average latency
rate(nexuszero_request_latency_seconds_sum[5m]) / rate(nexuszero_request_latency_seconds_count[5m])
```

### Error Metrics

#### `nexuszero_error_count_total`
- **Type:** Counter
- **Description:** Total number of errors
- **Labels:**
  - `service`: Service name
  - `error_type`: Type of error (validation, timeout, internal, etc.)
  - `severity`: Error severity (warning, error, critical)

**Example:**
```promql
# Error rate by type
rate(nexuszero_error_count_total[5m])

# Critical errors
sum(rate(nexuszero_error_count_total{severity="critical"}[5m])) by (service)
```

### System Metrics

#### `nexuszero_cpu_usage_percent`
- **Type:** Gauge
- **Description:** Current CPU usage percentage
- **Labels:**
  - `service`: Service name

#### `nexuszero_memory_usage_bytes`
- **Type:** Gauge
- **Description:** Current memory usage in bytes
- **Labels:**
  - `service`: Service name

#### `nexuszero_goroutines_count` / `nexuszero_threads_count`
- **Type:** Gauge
- **Description:** Number of active goroutines/threads
- **Labels:**
  - `service`: Service name

## Service-Specific Metrics

### Nexuszero Crypto Service

#### `nexuszero_crypto_proof_generation_duration_seconds`
- **Type:** Histogram
- **Description:** Time taken to generate cryptographic proofs
- **Labels:**
  - `proof_type`: Type of proof (ring-lwe, schnorr, etc.)

#### `nexuszero_crypto_verification_duration_seconds`
- **Type:** Histogram
- **Description:** Time taken to verify proofs
- **Labels:**
  - `proof_type`: Type of proof

#### `nexuszero_crypto_key_generation_total`
- **Type:** Counter
- **Description:** Total number of keys generated
- **Labels:**
  - `key_type`: Type of key (public, private, symmetric)

### Nexuszero Optimizer Service

#### `nexuszero_optimizer_training_loss`
- **Type:** Gauge
- **Description:** Current training loss value
- **Labels:**
  - `model`: Model name

#### `nexuszero_optimizer_training_accuracy`
- **Type:** Gauge
- **Description:** Current training accuracy
- **Labels:**
  - `model`: Model name

#### `nexuszero_optimizer_inference_duration_seconds`
- **Type:** Histogram
- **Description:** Time taken for model inference
- **Labels:**
  - `model`: Model name

#### `nexuszero_optimizer_dataset_size`
- **Type:** Gauge
- **Description:** Current dataset size
- **Labels:**
  - `dataset_type`: Type of dataset (training, validation, test)

### Nexuszero Monitor Service

#### `nexuszero_monitor_alerts_total`
- **Type:** Counter
- **Description:** Total number of alerts triggered
- **Labels:**
  - `alert_type`: Type of alert
  - `severity`: Alert severity

#### `nexuszero_monitor_health_checks_total`
- **Type:** Counter
- **Description:** Total number of health checks performed
- **Labels:**
  - `target`: Target service
  - `status`: Health check status (healthy, unhealthy)

## Implementation Guide

### Rust Services (using `prometheus` crate)

```rust
use prometheus::{
    Counter, Histogram, HistogramOpts, Opts, Registry, 
    TextEncoder, Encoder
};
use warp::Filter;

// Define metrics
lazy_static! {
    static ref REQUEST_COUNTER: Counter = Counter::new(
        "nexuszero_request_count_total",
        "Total number of requests"
    ).unwrap();
    
    static ref REQUEST_LATENCY: Histogram = Histogram::with_opts(
        HistogramOpts::new(
            "nexuszero_request_latency_seconds",
            "Request latency in seconds"
        )
        .buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0])
    ).unwrap();
}

// Register metrics
fn register_metrics() -> Registry {
    let registry = Registry::new();
    registry.register(Box::new(REQUEST_COUNTER.clone())).unwrap();
    registry.register(Box::new(REQUEST_LATENCY.clone())).unwrap();
    registry
}

// Metrics endpoint
async fn metrics_handler() -> Result<impl warp::Reply, warp::Rejection> {
    let registry = register_metrics();
    let encoder = TextEncoder::new();
    let mut buffer = vec![];
    let metric_families = registry.gather();
    encoder.encode(&metric_families, &mut buffer).unwrap();
    
    Ok(warp::reply::with_header(
        buffer,
        "content-type",
        encoder.format_type(),
    ))
}

// Add to your routes
let metrics_route = warp::path("metrics").and_then(metrics_handler);
```

### Python Services (using `prometheus-client`)

```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from flask import Flask, Response

app = Flask(__name__)

# Define metrics
REQUEST_COUNT = Counter(
    'nexuszero_request_count_total',
    'Total number of requests',
    ['service', 'method', 'endpoint', 'status_code']
)

REQUEST_LATENCY = Histogram(
    'nexuszero_request_latency_seconds',
    'Request latency in seconds',
    ['service', 'method', 'endpoint'],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
)

TRAINING_LOSS = Gauge(
    'nexuszero_optimizer_training_loss',
    'Current training loss',
    ['model']
)

@app.route('/metrics')
def metrics():
    return Response(generate_latest(), mimetype='text/plain')

# Usage
REQUEST_COUNT.labels(
    service='nexuszero-optimizer',
    method='POST',
    endpoint='/train',
    status_code='200'
).inc()

with REQUEST_LATENCY.labels(
    service='nexuszero-optimizer',
    method='POST',
    endpoint='/train'
).time():
    # Your code here
    pass
```

## Grafana Queries

### Example Queries

**Request Rate:**
```promql
sum(rate(nexuszero_request_count_total[5m])) by (service)
```

**Error Rate Percentage:**
```promql
100 * sum(rate(nexuszero_request_count_total{status_code=~"5.."}[5m])) by (service)
/ sum(rate(nexuszero_request_count_total[5m])) by (service)
```

**P99 Latency:**
```promql
histogram_quantile(0.99, 
  sum(rate(nexuszero_request_latency_seconds_bucket[5m])) by (service, le)
)
```

**Memory Usage:**
```promql
nexuszero_memory_usage_bytes / 1024 / 1024
```

## Alerts

Example Prometheus alerting rules:

```yaml
groups:
  - name: nexuszero
    interval: 30s
    rules:
      - alert: HighErrorRate
        expr: |
          sum(rate(nexuszero_error_count_total[5m])) by (service) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate on {{ $labels.service }}"
          
      - alert: HighLatency
        expr: |
          histogram_quantile(0.99, 
            sum(rate(nexuszero_request_latency_seconds_bucket[5m])) by (service, le)
          ) > 1.0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High latency on {{ $labels.service }}"
```

## Testing Metrics

Test metrics locally:

```bash
# Check metrics endpoint
curl http://localhost:13001/metrics

# Query Prometheus
curl 'http://localhost:9090/api/v1/query?query=nexuszero_request_count_total'
```

## Best Practices

1. **Use labels wisely**: Don't use high-cardinality labels (like user IDs)
2. **Consistent naming**: Follow the pattern `service_subsystem_metric_unit`
3. **Document metrics**: Keep this document updated with new metrics
4. **Set appropriate buckets**: For histograms, choose buckets that match your SLOs
5. **Monitor cardinality**: Too many label combinations can overwhelm Prometheus

## References

- [Prometheus Best Practices](https://prometheus.io/docs/practices/naming/)
- [Prometheus Metric Types](https://prometheus.io/docs/concepts/metric_types/)
- [Grafana Dashboard Best Practices](https://grafana.com/docs/grafana/latest/dashboards/build-dashboards/best-practices/)
