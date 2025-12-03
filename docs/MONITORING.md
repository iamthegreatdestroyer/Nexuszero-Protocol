# NexusZero ZK System Monitoring

This document describes the comprehensive monitoring infrastructure for the NexusZero Zero-Knowledge proof system.

## Overview

The monitoring system provides observability across four key areas:

1. **Proof Generation Metrics** - Timing, success rates, and resource usage
2. **Verification Latency** - End-to-end verification performance
3. **Circuit Compilation Times** - Circuit building and optimization metrics
4. **Error Tracking** - Structured logging with severity classification

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     ZK System Observability                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌─────────────┐    ┌────────────────────────┐  │
│  │   ZK Proof   │    │  Prometheus │    │   Grafana Dashboard    │  │
│  │   System     │───▶│   Scraper   │───▶│   nexuszero-zk-        │  │
│  │              │    │             │    │   monitoring           │  │
│  └──────────────┘    └─────────────┘    └────────────────────────┘  │
│         │                  │                                        │
│         │                  ▼                                        │
│         │           ┌─────────────┐                                 │
│         │           │ AlertManager│───▶ PagerDuty/Slack/Email      │
│         │           └─────────────┘                                 │
│         │                                                           │
│         ▼                                                           │
│  ┌──────────────┐    ┌─────────────┐    ┌────────────────────────┐  │
│  │   Error      │    │    Loki     │    │   Grafana Explore      │  │
│  │   Tracker    │───▶│   (Logs)    │───▶│   (Log Analysis)       │  │
│  │              │    │             │    │                        │  │
│  └──────────────┘    └─────────────┘    └────────────────────────┘  │
│         │                                                           │
│         ▼                                                           │
│  ┌──────────────┐                                                   │
│  │   Sentry     │───▶ Error Aggregation & Alerting                 │
│  │  (Optional)  │                                                   │
│  └──────────────┘                                                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. ZK Metrics Module (`nexuszero-crypto/src/metrics/zk_metrics.rs`)

Core metrics collection for ZK operations:

```rust
use nexuszero_crypto::metrics::{ZkMetrics, ProofType, SecurityLevel};

// Initialize metrics (call once at startup)
ZkMetrics::init().expect("Failed to initialize metrics");

// Get the global metrics instance
let metrics = ZkMetrics::global();

// Record proof generation
let guard = metrics.start_proof_generation(
    ProofType::Groth16,
    SecurityLevel::Bit128
);
// ... perform proof generation ...
guard.finish_success(1024, 1_000_000); // proof_size, constraints

// Record verification
metrics.record_verification(
    50.0,  // duration_ms
    true,  // success
    ProofType::Groth16,
    SecurityLevel::Bit128
);

// Record circuit compilation
metrics.record_circuit_compilation(
    200.0,   // duration_ms
    10000,   // complexity (gates)
    true,    // success
);
```

#### Available Metrics

| Metric                                    | Type      | Labels                             | Description              |
| ----------------------------------------- | --------- | ---------------------------------- | ------------------------ |
| `zk_proof_generation_duration_seconds`    | Histogram | proof_type, security_level         | Proof generation time    |
| `zk_proofs_generated_total`               | Counter   | proof_type, security_level, status | Total proofs generated   |
| `zk_proof_size_bytes`                     | Histogram | proof_type                         | Proof size distribution  |
| `zk_verification_duration_seconds`        | Histogram | proof_type, security_level         | Verification time        |
| `zk_verifications_total`                  | Counter   | proof_type, security_level, status | Total verifications      |
| `zk_circuit_compilation_duration_seconds` | Histogram | -                                  | Circuit compilation time |
| `zk_circuit_complexity`                   | Histogram | -                                  | Circuit gate count       |
| `zk_memory_usage_bytes`                   | Gauge     | operation                          | Memory consumption       |
| `zk_concurrent_operations`                | Gauge     | operation_type                     | Active operations        |
| `zk_errors_total`                         | Counter   | category, severity                 | Error counts             |
| `zk_sla_breaches_total`                   | Counter   | operation, target_latency_ms       | SLA violations           |

### 2. Error Tracking (`nexuszero-crypto/src/metrics/error_tracking.rs`)

Structured error logging with Sentry integration:

```rust
use nexuszero_crypto::metrics::{
    ZkErrorTracker, ErrorContext, ErrorSeverity, StructuredError
};

// Create error tracker
let tracker = ZkErrorTracker::new();

// Track an error
tracker.track_error(StructuredError {
    category: "proof_generation".to_string(),
    message: "Failed to generate witness".to_string(),
    severity: ErrorSeverity::Error,
    context: ErrorContext {
        operation_id: Some(uuid::Uuid::new_v4().to_string()),
        circuit_id: Some("groth16_v1".to_string()),
        proof_type: Some("groth16".to_string()),
        ..Default::default()
    },
    stack_trace: None,
    timestamp: chrono::Utc::now(),
});

// Get error summary
let summary = tracker.get_summary();
println!("Total errors: {}", summary.total_errors);
```

#### Error Categories

| Category              | Description                  |
| --------------------- | ---------------------------- |
| `proof_generation`    | Proof creation failures      |
| `verification`        | Proof verification failures  |
| `circuit_compilation` | Circuit building errors      |
| `witness_generation`  | Witness computation errors   |
| `parameter_error`     | Invalid security parameters  |
| `memory_error`        | Memory allocation failures   |
| `timeout`             | Operation timeouts           |
| `security_violation`  | Security constraint breaches |

### 3. HTTP Metrics Server (`nexuszero-crypto/src/metrics/http_server.rs`)

Exposes metrics for Prometheus scraping:

```rust
use nexuszero_crypto::metrics::{MetricsServer, spawn_metrics_server};

// Option 1: Run server blocking
let server = MetricsServer::from_str("0.0.0.0:13001")?;
server.run().await?;

// Option 2: Spawn in background
let handle = spawn_metrics_server("0.0.0.0:13001");
```

#### Endpoints

| Path       | Method | Description        |
| ---------- | ------ | ------------------ |
| `/metrics` | GET    | Prometheus metrics |
| `/health`  | GET    | Health check       |
| `/ready`   | GET    | Readiness check    |

## Grafana Dashboard

Import the dashboard from `grafana/dashboards/nexuszero-zk-monitoring.json`.

### Dashboard Panels

#### Row 1: Overview

- **Proofs Generated** - Total proof count
- **Active Generations** - Current in-progress proofs
- **Verifications/min** - Verification throughput
- **SLA Breaches** - SLA violation count

#### Row 2: Latency

- **Proof Generation Latency** - p50, p95, p99 percentiles
- **Verification Latency** - Time series trend

#### Row 3: Circuits

- **Circuit Compilation Time** - Build duration
- **Memory Usage** - Current memory consumption

#### Row 4: Errors

- **Error Rate by Category** - Stacked area chart
- **Resource Usage** - CPU, memory, disk

#### Row 5: Distribution

- **Proof Size Distribution** - Histogram
- **Security Level Distribution** - Pie chart

### Dashboard Variables

| Variable      | Description                                  |
| ------------- | -------------------------------------------- |
| `datasource`  | Prometheus data source                       |
| `job`         | Service job filter                           |
| `proof_type`  | Filter by proof type (groth16, plonk, stark) |
| `environment` | Environment filter (dev, staging, prod)      |

## Alerting Rules

Alerting rules are defined in `alerts/zk-system-alerts.yml`.

### Alert Groups

#### zk-proof-generation-alerts

| Alert                    | Severity | Condition               |
| ------------------------ | -------- | ----------------------- |
| ZkHighGenerationLatency  | warning  | p99 > 5s for 5m         |
| ZkFailedGenerations      | critical | >5% failure rate for 5m |
| ZkGenerationQueueBacklog | warning  | >10 concurrent for 10m  |
| ZkHighMemoryUsage        | warning  | >80% memory for 5m      |

#### zk-verification-alerts

| Alert                     | Severity | Condition               |
| ------------------------- | -------- | ----------------------- |
| ZkHighVerificationLatency | warning  | p99 > 500ms for 5m      |
| ZkVerificationFailures    | critical | >1% failure rate for 5m |
| ZkVerificationSLABreach   | warning  | any breach in 5m        |

#### zk-circuit-alerts

| Alert                    | Severity | Condition        |
| ------------------------ | -------- | ---------------- |
| ZkCircuitCompilationSlow | warning  | p95 > 30s for 5m |
| ZkCircuitComplexityHigh  | warning  | avg > 1M gates   |
| ZkCircuitCacheHitRateLow | warning  | <50% cache hits  |

#### zk-error-alerts

| Alert                    | Severity | Condition                |
| ------------------------ | -------- | ------------------------ |
| ZkHighErrorRate          | warning  | >1% error rate for 5m    |
| ZkCriticalErrors         | critical | any critical error in 5m |
| ZkMemoryErrors           | critical | memory errors in 5m      |
| ZkSecurityParamViolation | critical | any security violation   |

#### zk-sla-alerts

| Alert                   | Severity | Condition                |
| ----------------------- | -------- | ------------------------ |
| ZkGenerationSLABreach   | warning  | >10 breaches in 5m       |
| ZkVerificationSLABreach | warning  | any breach in 5m         |
| ZkOverallSLABreach      | critical | >50 total breaches in 5m |

## Configuration

### Prometheus Configuration

Add to your `prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

alerting:
  alertmanagers:
    - static_configs:
        - targets: ["alertmanager:9093"]

rule_files:
  - "alerts/*.yml"

scrape_configs:
  - job_name: "nexuszero-crypto"
    static_configs:
      - targets: ["nexuszero-crypto:13001"]
    metrics_path: "/metrics"
    scrape_interval: 10s
```

### Docker Compose Integration

```yaml
services:
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - ./alerts:/etc/prometheus/alerts
    ports:
      - "9090:9090"

  alertmanager:
    image: prom/alertmanager:latest
    ports:
      - "9093:9093"

  grafana:
    image: grafana/grafana:latest
    volumes:
      - ./grafana/dashboards:/var/lib/grafana/dashboards
      - ./grafana/provisioning:/etc/grafana/provisioning
    ports:
      - "3000:3000"
```

## Usage Examples

### Recording Metrics in Application Code

```rust
use nexuszero_crypto::metrics::{ZkMetrics, ProofType, SecurityLevel};

pub async fn generate_proof(
    circuit: &Circuit,
    witness: &Witness,
) -> Result<Proof, Error> {
    let metrics = ZkMetrics::global();

    // Start timing
    let guard = metrics.start_proof_generation(
        ProofType::Groth16,
        SecurityLevel::Bit128,
    );

    // Record memory before
    let mem_before = get_memory_usage();
    metrics.record_memory_usage("proof_generation", mem_before);

    // Generate proof
    let result = circuit.prove(witness).await;

    // Record result
    match &result {
        Ok(proof) => {
            let proof_size = proof.serialized_size();
            let constraints = circuit.constraint_count();
            guard.finish_success(proof_size, constraints);
        }
        Err(e) => {
            guard.finish_failure(&e.to_string());
            metrics.record_error(
                "proof_generation",
                "error",
                &e.to_string(),
            );
        }
    }

    result
}
```

### SLA Monitoring

```rust
// Define SLA targets
const PROOF_GENERATION_SLA_MS: f64 = 5000.0;
const VERIFICATION_SLA_MS: f64 = 100.0;

// Check and record SLA breaches
if duration_ms > PROOF_GENERATION_SLA_MS {
    metrics.record_sla_breach(
        "proof_generation",
        PROOF_GENERATION_SLA_MS as u64,
    );
}
```

### Error Tracking with Context

```rust
use nexuszero_crypto::metrics::{ZkErrorTracker, ErrorContext, ErrorSeverity};

let tracker = ZkErrorTracker::new();

// Track error with full context
tracker.track_error(StructuredError {
    category: "verification".to_string(),
    message: format!("Verification failed: invalid public input"),
    severity: ErrorSeverity::Warning,
    context: ErrorContext {
        operation_id: Some(op_id.to_string()),
        circuit_id: Some(circuit_id.to_string()),
        proof_type: Some("plonk".to_string()),
        security_level: Some(128),
        constraint_count: Some(100000),
        memory_usage_bytes: Some(1024 * 1024 * 100),
        duration_ms: Some(150.5),
        custom_fields: [
            ("input_hash".to_string(), "0xabc...".to_string()),
            ("expected_hash".to_string(), "0xdef...".to_string()),
        ].into_iter().collect(),
    },
    stack_trace: Some(backtrace::Backtrace::capture().to_string()),
    timestamp: chrono::Utc::now(),
});
```

## Best Practices

### 1. Metric Naming

- Use snake_case for metric names
- Prefix with `zk_` for ZK-specific metrics
- Include units in metric name (e.g., `_seconds`, `_bytes`)

### 2. Label Cardinality

- Keep label cardinality low to avoid metric explosion
- Use bounded enums (ProofType, SecurityLevel)
- Avoid using unbounded IDs as labels

### 3. Histogram Buckets

- Choose buckets based on expected latency distribution
- For proof generation: 0.1s to 30s
- For verification: 10ms to 1s

### 4. Error Handling

- Always track errors with context
- Use appropriate severity levels
- Include operation IDs for correlation

### 5. SLA Tracking

- Define clear SLA targets
- Monitor breach frequency
- Alert before reaching critical thresholds

## Troubleshooting

### No Metrics Appearing

1. Check if metrics server is running: `curl http://localhost:13001/health`
2. Verify Prometheus scrape config
3. Check for initialization errors in logs

### High Memory Usage

1. Check for metric cardinality explosion
2. Review label usage
3. Consider reducing histogram buckets

### Missing Alerts

1. Verify `rule_files` path in prometheus.yml
2. Check AlertManager configuration
3. Review alert expressions in Grafana

## References

- [Prometheus Best Practices](https://prometheus.io/docs/practices/)
- [Grafana Dashboard Guidelines](https://grafana.com/docs/grafana/latest/best-practices/)
- [Alerting Best Practices](https://prometheus.io/docs/practices/alerting/)
