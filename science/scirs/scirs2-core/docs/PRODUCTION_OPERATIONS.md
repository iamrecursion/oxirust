# SciRS2 Core 1.0 Production Operations Guide

## Table of Contents

1. [Operations Overview](#operations-overview)
2. [Monitoring Setup](#monitoring-setup)
3. [Alerting and Incident Response](#alerting-and-incident-response)
4. [Performance Tuning](#performance-tuning)
5. [Capacity Planning](#capacity-planning)
6. [Disaster Recovery](#disaster-recovery)
7. [Security Operations](#security-operations)
8. [Compliance and Auditing](#compliance-and-auditing)
9. [Runbooks](#runbooks)
10. [SLA and KPI Management](#sla-and-kpi-management)

## Operations Overview

This guide provides operational procedures for managing SciRS2 Core 1.0 in production environments. It complements the [Production Deployment Guide](PRODUCTION_DEPLOYMENT.md) with day-to-day operational procedures, monitoring, and incident response.

### Service Level Objectives (SLOs)

#### Availability SLOs
- **Uptime**: 99.9% (8.77 hours downtime per year)
- **Response Time**: 95th percentile < 500ms for standard operations
- **Throughput**: Maintain >1000 operations/second under normal load

#### Performance SLOs
- **Memory Efficiency**: <90% memory utilization under normal load
- **CPU Utilization**: <80% average CPU usage
- **Error Rate**: <0.1% for all operations

#### Recovery SLOs
- **RTO (Recovery Time Objective)**: 15 minutes
- **RPO (Recovery Point Objective)**: 1 hour for non-critical data

## Monitoring Setup

### Core Metrics Collection

#### Application Metrics
```rust
use scirs2_core::metrics::{MetricType, global_metrics_registry};

// Performance metrics
let registry = global_metrics_registry();

// Operation counters
registry.counter("scirs2_operations_total", "Total operations performed")
    .with_labels(&[("operation", "matrix_multiply"), ("status", "success")]);

// Latency histograms
registry.histogram("scirs2_operation_duration_seconds", "Operation duration")
    .with_buckets(&[0.001, 0.01, 0.1, 1.0, 10.0]);

// Resource utilization gauges
registry.gauge("scirs2_memory_usage_bytes", "Current memory usage");
registry.gauge("scirs2_cpu_usage_percent", "Current CPU usage");
registry.gauge("scirs2_gpu_memory_usage_bytes", "Current GPU memory usage");

// Error tracking
registry.counter("scirs2_errors_total", "Total errors")
    .with_labels(&[("error_type", "computation"), ("severity", "high")]);
```

#### System Metrics
```bash
# CPU metrics
node_cpu_seconds_total
node_load1, node_load5, node_load15

# Memory metrics
node_memory_MemTotal_bytes
node_memory_MemAvailable_bytes
node_memory_MemFree_bytes

# Disk metrics
node_filesystem_size_bytes
node_filesystem_avail_bytes
node_disk_io_time_seconds_total

# Network metrics
node_network_receive_bytes_total
node_network_transmit_bytes_total
```

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "scirs2_alerts.yml"

scrape_configs:
  - job_name: 'scirs2-core'
    static_configs:
      - targets: ['localhost:9090']
    metrics_path: '/metrics'
    scrape_interval: 5s
    
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['localhost:9093']
```

### Grafana Dashboard Configuration

```json
{
  "dashboard": {
    "title": "SciRS2 Core Production Dashboard",
    "panels": [
      {
        "title": "Operations per Second",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(scirs2_operations_total[1m])",
            "legendFormat": "{{operation}}"
          }
        ]
      },
      {
        "title": "Response Time P95",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(scirs2_operation_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "scirs2_memory_usage_bytes / 1024 / 1024 / 1024",
            "legendFormat": "Memory (GB)"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(scirs2_errors_total[5m])",
            "legendFormat": "{{error_type}}"
          }
        ]
      }
    ]
  }
}
```

### Distributed Tracing Setup

```rust
use scirs2_core::observability::tracing::{TracingConfig, JaegerExporter};

// Configure distributed tracing
let tracing_config = TracingConfig::new()
    .with_service_name("scirs2-core")
    .with_service_version("1.0.0")
    .with_environment("production")
    .with_sampling_rate(0.1); // 10% sampling in production

// Export to Jaeger
let jaeger_exporter = JaegerExporter::new("http://jaeger:14268/api/traces")?;
jaeger_exporter.install_global(tracing_config)?;

// Trace operations
use scirs2_core::observability::tracing::{trace_span, Span};

fn compute_matrix_multiplication(a: &Array2<f64>, b: &Array2<f64>) -> CoreResult<Array2<f64>> {
    let _span = trace_span!("matrix_multiplication")
        .with_attribute("matrix_a_shape", format!("{:?}", a.shape()))
        .with_attribute("matrix_b_shape", format!("{:?}", b.shape()));
    
    // Computation logic here
    Ok(result)
}
```

## Alerting and Incident Response

### Alert Rules

```yaml
# scirs2_alerts.yml
groups:
  - name: scirs2.critical
    rules:
      - alert: SciRS2ServiceDown
        expr: up{job="scirs2-core"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "SciRS2 service is down"
          description: "SciRS2 service has been down for more than 1 minute"

      - alert: SciRS2HighErrorRate
        expr: rate(scirs2_errors_total[5m]) > 0.01
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High error rate in SciRS2"
          description: "Error rate is {{ $value }} errors/sec for 2 minutes"

      - alert: SciRS2HighLatency
        expr: histogram_quantile(0.95, rate(scirs2_operation_duration_seconds_bucket[5m])) > 1.0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High latency in SciRS2 operations"
          description: "95th percentile latency is {{ $value }}s"

      - alert: SciRS2HighMemoryUsage
        expr: scirs2_memory_usage_bytes / node_memory_MemTotal_bytes > 0.9
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage in SciRS2"
          description: "Memory usage is {{ $value | humanizePercentage }}"

      - alert: SciRS2GPUMemoryExhaustion
        expr: scirs2_gpu_memory_usage_bytes / scirs2_gpu_memory_total_bytes > 0.95
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "GPU memory exhaustion"
          description: "GPU memory usage is {{ $value | humanizePercentage }}"
```

### Incident Response Playbook

#### High Error Rate Incident

**Severity**: Critical  
**Expected Response Time**: 5 minutes

**Steps**:
1. **Immediate Response** (0-5 minutes)
   ```bash
   # Check service status
   curl -f http://localhost:8080/health
   
   # Check recent logs
   journalctl -u scirs2-service --since "5 minutes ago"
   
   # Check resource usage
   top -p $(pgrep scirs2)
   ```

2. **Investigation** (5-15 minutes)
   ```bash
   # Analyze error patterns
   grep -E "(ERROR|FATAL)" /var/log/scirs2/app.log | tail -50
   
   # Check system resources
   free -h
   df -h
   iostat -x 1 5
   
   # Review recent deployments
   git log --oneline --since="1 hour ago"
   ```

3. **Mitigation** (15-30 minutes)
   ```bash
   # Scale horizontally if possible
   kubectl scale deployment scirs2-core --replicas=5
   
   # Restart service if memory leak suspected
   systemctl restart scirs2-service
   
   # Route traffic to healthy instances
   kubectl patch service scirs2-core -p '{"spec":{"selector":{"health":"healthy"}}}'
   ```

#### Memory Exhaustion Incident

**Severity**: Warning → Critical  
**Expected Response Time**: 10 minutes

**Steps**:
1. **Assessment**
   ```rust
   use scirs2_core::memory::metrics::{MemorySnapshot, take_snapshot};
   
   // Take memory snapshot
   let snapshot = take_snapshot();
   println!("Memory breakdown: {:#?}", snapshot);
   
   // Check for memory leaks
   let leak_detector = LeakDetector::new();
   let leaks = leak_detector.detect_leaks()?;
   if !leaks.is_empty() {
       eprintln!("Memory leaks detected: {:?}", leaks);
   }
   ```

2. **Immediate Actions**
   ```bash
   # Force garbage collection if applicable
   kill -USR1 $(pgrep scirs2)  # Custom signal for memory cleanup
   
   # Reduce memory pressure
   echo 3 > /proc/sys/vm/drop_caches
   
   # Scale down non-critical processes
   systemctl stop scirs2-background-tasks
   ```

3. **Recovery**
   ```bash
   # Enable memory-efficient mode
   export SCIRS2_MEMORY_EFFICIENT=true
   export SCIRS2_CHUNK_SIZE=524288  # Reduce chunk size
   
   # Restart with lower memory profile
   systemctl restart scirs2-service
   ```

### On-Call Procedures

#### Alert Escalation Matrix
| Severity | Initial Response | Escalation (15 min) | Escalation (30 min) |
|----------|------------------|---------------------|---------------------|
| Critical | Primary On-Call  | Senior Engineer     | Engineering Manager |
| Warning  | Primary On-Call  | -                   | Senior Engineer     |
| Info     | Automated Ticket | -                   | -                   |

#### Communication Channels
- **Incident Command**: Slack #scirs2-incidents
- **Status Page**: https://status.scirs2.org
- **Stakeholder Updates**: Email to stakeholders@company.com

## Performance Tuning

### CPU Optimization

```rust
use scirs2_core::resource::{get_system_resources, optimize_cpu_affinity};

// Detect optimal CPU configuration
let resources = get_system_resources();
println!("CPU cores: {}", resources.cpu_cores);
println!("NUMA nodes: {}", resources.numa_nodes);

// Optimize CPU affinity for NUMA
optimize_cpu_affinity(&resources)?;

// Configure thread pool
use scirs2_core::parallel_ops::set_num_threads;
set_num_threads(resources.cpu_cores.min(16)); // Cap at 16 threads
```

### Memory Optimization

```rust
use scirs2_core::memory::{BufferPool, MemoryConfig};

// Configure memory pools
let memory_config = MemoryConfig::new()
    .with_pool_size(1_073_741_824)  // 1GB pool
    .with_chunk_size(1_048_576)     // 1MB chunks
    .with_alignment(64)             // 64-byte alignment for SIMD
    .with_numa_aware(true);

let buffer_pool = BufferPool::new(memory_config)?;

// Enable huge pages for large allocations
use scirs2_core::memory::enable_huge_pages;
enable_huge_pages()?;
```

### GPU Optimization

```rust
use scirs2_core::gpu::{GpuConfig, GpuOptimizer};

// Optimize GPU configuration
let gpu_config = GpuConfig::new()
    .with_memory_pool_size(4_294_967_296)  // 4GB GPU memory pool
    .with_concurrent_streams(4)
    .with_tensor_core_optimization(true);

let gpu_optimizer = GpuOptimizer::new(gpu_config)?;
gpu_optimizer.apply_optimizations()?;
```

### Network Optimization

```bash
# Optimize network settings for distributed computing
# Increase network buffer sizes
echo 'net.core.rmem_max = 268435456' >> /etc/sysctl.conf
echo 'net.core.wmem_max = 268435456' >> /etc/sysctl.conf
echo 'net.ipv4.tcp_rmem = 4096 87380 268435456' >> /etc/sysctl.conf
echo 'net.ipv4.tcp_wmem = 4096 65536 268435456' >> /etc/sysctl.conf

# Apply settings
sysctl -p
```

## Capacity Planning

### Resource Usage Forecasting

```rust
use scirs2_core::metrics::{MetricsCollector, ResourcePredictor};

// Collect historical metrics
let collector = MetricsCollector::new();
let historical_data = collector.collect_range(
    chrono::Utc::now() - chrono::Duration::days(30),
    chrono::Utc::now()
)?;

// Predict future resource needs
let predictor = ResourcePredictor::new();
let forecast = predictor.predict_resource_usage(
    &historical_data,
    chrono::Duration::days(90)  // 90-day forecast
)?;

println!("Projected CPU usage: {:.2}%", forecast.cpu_usage_percent);
println!("Projected memory usage: {:.2} GB", forecast.memory_usage_gb);
println!("Projected storage needs: {:.2} TB", forecast.storage_usage_tb);
```

### Scaling Triggers

#### Horizontal Scaling
```yaml
# Kubernetes HPA configuration
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: scirs2-core-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: scirs2-core
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: scirs2_operations_per_second
      target:
        type: AverageValue
        averageValue: "100"
```

#### Vertical Scaling
```rust
use scirs2_core::resource::{ResourceMonitor, ScalingRecommendation};

// Monitor resource usage and get scaling recommendations
let monitor = ResourceMonitor::new();
let recommendation = monitor.get_scaling_recommendation()?;

match recommendation {
    ScalingRecommendation::ScaleUp { cpu_cores, memory_gb } => {
        println!("Recommend scaling up: {} cores, {} GB RAM", cpu_cores, memory_gb);
    },
    ScalingRecommendation::ScaleDown { cpu_cores, memory_gb } => {
        println!("Can scale down: {} cores, {} GB RAM", cpu_cores, memory_gb);
    },
    ScalingRecommendation::NoChange => {
        println!("Current resources are optimal");
    }
}
```

### Cost Optimization

```rust
use scirs2_core::resource::{CostOptimizer, ResourceUtilization};

// Analyze resource utilization for cost optimization
let cost_optimizer = CostOptimizer::new();
let utilization = ResourceUtilization::collect_current()?;

let optimizations = cost_optimizer.recommend_optimizations(&utilization)?;
for optimization in optimizations {
    println!("Cost saving opportunity: {}", optimization.description);
    println!("Estimated savings: ${:.2}/month", optimization.monthly_savings);
}
```

## Disaster Recovery

### Backup Procedures

#### Configuration Backup
```bash
#!/bin/bash
# backup-config.sh

BACKUP_DIR="/backup/scirs2/$(date +%Y%m%d)"
mkdir -p "$BACKUP_DIR"

# Backup configuration files
tar -czf "$BACKUP_DIR/config.tar.gz" \
    /etc/scirs2/ \
    ~/.config/scirs2/ \
    /var/lib/scirs2/config/

# Backup application state
tar -czf "$BACKUP_DIR/state.tar.gz" \
    /var/lib/scirs2/state/ \
    /var/lib/scirs2/cache/

# Backup logs (last 7 days)
find /var/log/scirs2/ -mtime -7 -type f | \
    tar -czf "$BACKUP_DIR/logs.tar.gz" -T -

# Backup metrics and monitoring data
tar -czf "$BACKUP_DIR/metrics.tar.gz" \
    /var/lib/prometheus/ \
    /var/lib/grafana/
```

#### Data Backup
```rust
use scirs2_core::io::{BackupManager, BackupConfig};

// Configure automated backups
let backup_config = BackupConfig::new()
    .with_destination("s3://scirs2-backups/")
    .with_compression(true)
    .with_encryption(true)
    .with_retention_days(90)
    .with_incremental(true);

let backup_manager = BackupManager::new(backup_config)?;

// Perform backup
backup_manager.backup_data_directory("/var/lib/scirs2/data/")?;
backup_manager.backup_configuration("/etc/scirs2/")?;
```

### Recovery Procedures

#### Point-in-Time Recovery
```bash
#!/bin/bash
# restore-to-point.sh

RESTORE_POINT="2025-06-29T10:30:00Z"
BACKUP_DIR="/backup/scirs2"

# Stop services
systemctl stop scirs2-service

# Restore configuration
tar -xzf "$BACKUP_DIR/config.tar.gz" -C /

# Restore data to specific point in time
scirs2-restore --point-in-time "$RESTORE_POINT" \
               --source "$BACKUP_DIR" \
               --target "/var/lib/scirs2/"

# Verify integrity
scirs2-verify --data-directory "/var/lib/scirs2/data/"

# Start services
systemctl start scirs2-service

# Validate recovery
curl -f http://localhost:8080/health
```

#### Cross-Region Recovery
```rust
use scirs2_core::disaster_recovery::{CrossRegionReplication, FailoverManager};

// Configure cross-region replication
let replication = CrossRegionReplication::new()
    .with_primary_region("us-east-1")
    .with_secondary_regions(vec!["us-west-2", "eu-west-1"])
    .with_replication_lag_threshold(Duration::from_secs(30));

// Automatic failover configuration
let failover_manager = FailoverManager::new()
    .with_health_check_interval(Duration::from_secs(10))
    .with_failover_threshold(3)  // Fail after 3 consecutive failures
    .with_automatic_failback(true);

// Monitor and execute failover if needed
failover_manager.monitor_and_failover()?;
```

### Business Continuity Testing

```bash
#!/bin/bash
# disaster-recovery-test.sh

echo "Starting disaster recovery test..."

# Simulate primary region failure
echo "Simulating primary region failure..."
systemctl stop scirs2-service

# Test secondary region activation
echo "Activating secondary region..."
scirs2-failover --target-region "us-west-2"

# Validate services in secondary region
echo "Validating services..."
curl -f http://backup-cluster:8080/health

# Test data consistency
echo "Testing data consistency..."
scirs2-consistency-check --primary-backup --secondary-backup

# Simulate recovery
echo "Testing recovery to primary region..."
scirs2-failback --target-region "us-east-1"

echo "Disaster recovery test completed."
```

## Security Operations

### Security Monitoring

```rust
use scirs2_core::security::{SecurityMonitor, ThreatDetector};

// Initialize security monitoring
let security_monitor = SecurityMonitor::new()
    .with_anomaly_detection(true)
    .with_threat_intelligence(true)
    .with_compliance_monitoring(true);

// Detect threats
let threat_detector = ThreatDetector::new();
let threats = threat_detector.scan_for_threats()?;

for threat in threats {
    match threat.severity {
        ThreatSeverity::Critical => {
            // Immediate response required
            security_monitor.trigger_incident_response(&threat)?;
        },
        ThreatSeverity::High => {
            // Alert security team
            security_monitor.alert_security_team(&threat)?;
        },
        _ => {
            // Log for analysis
            security_monitor.log_threat(&threat)?;
        }
    }
}
```

### Access Control Monitoring

```rust
use scirs2_core::security::{AccessAudit, PermissionChecker};

// Audit access patterns
let access_audit = AccessAudit::new();
let suspicious_access = access_audit.detect_anomalous_access()?;

for access in suspicious_access {
    println!("Suspicious access detected: {:?}", access);
    
    // Check current permissions
    let permission_checker = PermissionChecker::new();
    let valid = permission_checker.validate_access(&access)?;
    
    if !valid {
        // Revoke access and alert
        access_audit.revoke_access(&access.user_id)?;
        access_audit.alert_security_incident(&access)?;
    }
}
```

### Vulnerability Management

```bash
#!/bin/bash
# vulnerability-scan.sh

echo "Starting vulnerability scan..."

# Scan dependencies for known vulnerabilities
cargo audit

# Scan container images
trivy image scirs2-core:1.0

# Scan infrastructure
nmap -sV -O localhost

# Generate security report
scirs2-security-report --output /var/log/scirs2/security-report.json

echo "Vulnerability scan completed."
```

## Compliance and Auditing

### Audit Logging

```rust
use scirs2_core::observability::audit::{AuditLogger, AuditEvent, ComplianceStandard};

// Configure audit logging for compliance
let audit_logger = AuditLogger::new()
    .with_compliance_standard(ComplianceStandard::SOX)
    .with_retention_period(chrono::Duration::days(2555))  // 7 years
    .with_encryption(true)
    .with_integrity_protection(true);

// Log critical operations
audit_logger.log(AuditEvent::new()
    .with_event_type("data_access")
    .with_user_id("user123")
    .with_resource("sensitive_dataset")
    .with_action("read")
    .with_outcome("success")
    .with_details("Accessed customer financial data for analysis")
)?;
```

### Compliance Reporting

```rust
use scirs2_core::compliance::{ComplianceReporter, ComplianceReport};

// Generate compliance reports
let reporter = ComplianceReporter::new();

// SOX compliance report
let sox_report = reporter.generate_sox_report(
    chrono::Utc::now() - chrono::Duration::days(90),
    chrono::Utc::now()
)?;

// GDPR compliance report
let gdpr_report = reporter.generate_gdpr_report(
    chrono::Utc::now() - chrono::Duration::days(30),
    chrono::Utc::now()
)?;

// Export reports
sox_report.export_to_file("/var/lib/scirs2/compliance/sox-report.pdf")?;
gdpr_report.export_to_file("/var/lib/scirs2/compliance/gdpr-report.pdf")?;
```

### Data Governance

```rust
use scirs2_core::governance::{DataClassifier, DataRetentionPolicy};

// Classify data automatically
let classifier = DataClassifier::new();
let classification = classifier.classify_dataset(&dataset)?;

// Apply retention policies
let retention_policy = DataRetentionPolicy::new()
    .with_classification_rules(vec![
        (DataClassification::Public, chrono::Duration::days(365)),
        (DataClassification::Internal, chrono::Duration::days(2555)),
        (DataClassification::Confidential, chrono::Duration::days(3650)),
        (DataClassification::Restricted, chrono::Duration::days(7300)),
    ]);

retention_policy.apply_to_dataset(&dataset, &classification)?;
```

## Runbooks

### Common Operational Tasks

#### Service Restart
```bash
#!/bin/bash
# service-restart.sh

SERVICE_NAME="scirs2-service"

echo "Performing graceful restart of $SERVICE_NAME..."

# Pre-restart health check
curl -f http://localhost:8080/health || echo "Service unhealthy before restart"

# Graceful shutdown
systemctl stop $SERVICE_NAME
sleep 10

# Verify shutdown
if pgrep -f $SERVICE_NAME; then
    echo "Force killing remaining processes..."
    pkill -f $SERVICE_NAME
fi

# Start service
systemctl start $SERVICE_NAME

# Post-restart validation
sleep 30
curl -f http://localhost:8080/health || {
    echo "Service failed to start properly"
    exit 1
}

echo "Service restart completed successfully"
```

#### Configuration Update
```bash
#!/bin/bash
# update-config.sh

CONFIG_FILE="/etc/scirs2/config.toml"
BACKUP_DIR="/backup/scirs2/config"

# Create backup
mkdir -p "$BACKUP_DIR"
cp "$CONFIG_FILE" "$BACKUP_DIR/config-$(date +%Y%m%d-%H%M%S).toml"

# Validate new configuration
scirs2-config-validate --file "$1" || {
    echo "Configuration validation failed"
    exit 1
}

# Apply new configuration
cp "$1" "$CONFIG_FILE"

# Reload configuration
systemctl reload scirs2-service

# Validate service health
sleep 10
curl -f http://localhost:8080/health || {
    echo "Service unhealthy after configuration update"
    # Rollback
    cp "$BACKUP_DIR/$(ls -t $BACKUP_DIR | head -1)" "$CONFIG_FILE"
    systemctl reload scirs2-service
    exit 1
}

echo "Configuration updated successfully"
```

#### Database Maintenance
```bash
#!/bin/bash
# database-maintenance.sh

echo "Starting database maintenance..."

# Backup database
scirs2-db-backup --output "/backup/scirs2/db-$(date +%Y%m%d).sql"

# Optimize database
scirs2-db-optimize --vacuum --reindex

# Update statistics
scirs2-db-analyze

# Check database health
scirs2-db-health-check || {
    echo "Database health check failed"
    exit 1
}

echo "Database maintenance completed"
```

## SLA and KPI Management

### Key Performance Indicators (KPIs)

#### Operational KPIs
```rust
use scirs2_core::metrics::{KpiCalculator, KpiMetric};

// Calculate operational KPIs
let kpi_calculator = KpiCalculator::new();

// Availability KPI
let availability = kpi_calculator.calculate_availability(
    chrono::Utc::now() - chrono::Duration::days(30),
    chrono::Utc::now()
)?;

// Performance KPIs
let performance_metrics = kpi_calculator.calculate_performance_metrics()?;

// Resource utilization KPIs
let utilization_metrics = kpi_calculator.calculate_utilization_metrics()?;

println!("=== SciRS2 Core KPI Report ===");
println!("Availability: {:.3}%", availability.percentage);
println!("Mean Response Time: {:.2}ms", performance_metrics.mean_response_time_ms);
println!("95th Percentile Response Time: {:.2}ms", performance_metrics.p95_response_time_ms);
println!("Error Rate: {:.4}%", performance_metrics.error_rate_percentage);
println!("CPU Utilization: {:.2}%", utilization_metrics.cpu_utilization_percentage);
println!("Memory Utilization: {:.2}%", utilization_metrics.memory_utilization_percentage);
```

#### Business KPIs
```rust
use scirs2_core::business_metrics::{BusinessKpiCalculator, UsageMetrics};

// Calculate business impact KPIs
let business_calculator = BusinessKpiCalculator::new();

// User engagement metrics
let engagement = business_calculator.calculate_user_engagement()?;

// Processing throughput metrics
let throughput = business_calculator.calculate_processing_throughput()?;

// Cost efficiency metrics
let cost_efficiency = business_calculator.calculate_cost_efficiency()?;

println!("=== Business KPI Report ===");
println!("Active Users: {}", engagement.active_users);
println!("Operations/Hour: {}", throughput.operations_per_hour);
println!("Cost/Operation: ${:.6}", cost_efficiency.cost_per_operation);
println!("Resource Efficiency: {:.2}%", cost_efficiency.resource_efficiency_percentage);
```

### SLA Reporting

```rust
use scirs2_core::sla::{SlaReporter, SlaMetrics, SlaTarget};

// Define SLA targets
let sla_targets = vec![
    SlaTarget::new("availability", 99.9),
    SlaTarget::new("response_time_p95", 500.0),  // milliseconds
    SlaTarget::new("error_rate", 0.1),          // percentage
];

// Generate SLA report
let sla_reporter = SlaReporter::new(sla_targets);
let sla_report = sla_reporter.generate_monthly_report(
    chrono::Utc::now().with_day(1).unwrap(),  // First day of month
    chrono::Utc::now()
)?;

// Check SLA compliance
for metric in &sla_report.metrics {
    let compliance_status = if metric.achieved >= metric.target {
        "✅ COMPLIANT"
    } else {
        "❌ NON-COMPLIANT"
    };
    
    println!("{}: {:.2}% (target: {:.2}%) - {}", 
             metric.name, 
             metric.achieved, 
             metric.target, 
             compliance_status);
}

// Generate executive summary
println!("\n=== Executive Summary ===");
println!("Overall SLA Compliance: {:.1}%", sla_report.overall_compliance_percentage);
println!("SLA Credits: ${:.2}", sla_report.sla_credits);
```

### Continuous Improvement

```rust
use scirs2_core::improvement::{PerformanceAnalyzer, ImprovementRecommendation};

// Analyze performance trends
let analyzer = PerformanceAnalyzer::new();
let trends = analyzer.analyze_trends(
    chrono::Utc::now() - chrono::Duration::days(90),
    chrono::Utc::now()
)?;

// Generate improvement recommendations
let recommendations = analyzer.generate_recommendations(&trends)?;

println!("=== Improvement Recommendations ===");
for rec in recommendations {
    println!("Priority {}: {}", rec.priority, rec.title);
    println!("  Description: {}", rec.description);
    println!("  Expected Impact: {}", rec.expected_impact);
    println!("  Implementation Effort: {}", rec.implementation_effort);
    println!();
}
```

---

**Version**: SciRS2 Core 1.0  
**Last Updated**: 2025-06-29  
**Authors**: SciRS2 Operations Team  
**License**: See LICENSE file for details