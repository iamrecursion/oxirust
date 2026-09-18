# oximedia-monitor TODO

## Current Status
- 48 source files/directories providing comprehensive monitoring, alerting, and observability
- Core: OximediaMonitor with MetricsCollector, SqliteStorage, QueryEngine, AlertManager
- Metrics: system (CPU, memory, disk, network), application (encoding, jobs, workers), quality (PSNR, SSIM, VMAF)
- Alerting: rules engine, pipeline, multi-channel (email, Slack, Discord, webhook, SMS, file)
- Storage: RingBuffer (in-memory) + SQLite (historical), retention policies
- Advanced: anomaly detection, correlation, SLA/SLO tracking, capacity planning, resource forecasting
- Features: default (none), gpu, integrations, email, sqlite, full
- Additional subdirs: audio, cc, compliance, focus, multiviewer, reference, scopes, status, technical, timecode

## Enhancements
- [x] Make the SQLite feature optional but still allow OximediaMonitor to work without it (in-memory only mode)
- [x] Add metric batching in MetricsCollector to reduce SQLite write frequency under high load
- [x] Implement alert deduplication in alerting_pipeline to suppress repeated firings within a cooldown window
- [x] Add exponential backoff to webhook/email alert delivery failures in AlertManager
- [x] Extend anomaly detection with seasonal decomposition (detect hourly/daily patterns in encoding throughput) (verified 2026-05-16; src/seasonal_decomposition.rs:1387 lines, src/seasonal.rs)
- [x] Add configurable metric aggregation granularity (1s, 10s, 1m, 5m) in metric_aggregation
- [x] Implement metric cardinality limits in metric_store to prevent unbounded label growth (verified 2026-05-16; src/cardinality_limiter.rs:709 lines, src/cardinality.rs)

## New Features
- [x] Add OpenTelemetry (OTLP) export support alongside existing Prometheus format in metric_export (verified 2026-05-16; src/opentelemetry_export.rs:678 lines)
- [x] Implement distributed tracing with W3C Trace Context propagation (extend trace_span module) (verified: trace_context.rs:TraceParent+TraceState, w3c_trace_context.rs:TraceState)
- [x] Add StatsD ingestion endpoint to accept metrics from external processes (verified 2026-05-16; src/statsd_ingestion.rs:1195 lines, src/statsd_parser.rs)
- [x] Implement metric recording/playback for debugging -- record live metrics, replay offline (verified: metric_recorder.rs:207:MetricRecorder, :422:PlaybackEvent, :499:next_event)
- [x] Add GPU metrics collection for non-NVIDIA GPUs (Apple Metal via IOKit, AMD via sysfs on Linux) (verified: gpu_metrics_sysfs.rs:180:AmdGpuReader, :462:AppleGpuMetrics, :595:GpuMetricsCollector)
- [x] Implement dashboard templating system with variable substitution in dashboard module (verified 2026-05-16; src/dashboard_template.rs:692 lines)
- [x] Add PagerDuty and OpsGenie integration to alerting channels (verified 2026-05-16; src/pagerduty.rs:696 lines, src/opsgenie.rs:764 lines)

## Performance
- [x] Replace SQLite writes with batch INSERT using transactions (collect N samples, write once) (verified: storage/sqlite.rs:insert_batch+unchecked_transaction)
- [x] Use dashmap more aggressively for hot-path metric lookups instead of Arc<Mutex<HashMap>> (verified: metrics/registry.rs:Arc<DashMap>, no Mutex<HashMap> in hot path)
- [x] Implement metric downsampling for historical data (keep 1s resolution for 1h, 1m for 24h, 5m for 30d) (verified: metric_downsample.rs:20:RetentionPolicy, :22:resolution_ms, multi-tier tiers vec)
- [x] Add connection pooling for SQLite to reduce open/close overhead in concurrent access (verified: sqlite_pool.rs:228:SqlitePool, :80:PoolConfig, PooledConnection)
- [x] Profile and optimize sysinfo collection -- skip metrics that are not enabled in config (verified: SystemMetricsConfig per-category gating in system.rs; with_config/new_with_options/new constructors; 4 new tests)

## Testing
- [x] Add integration test for the full alert pipeline: metric threshold -> rule evaluation -> notification dispatch (2026-06-22; wired AlertManager::evaluate_metric -- rule eval -> AlertChannel dispatch, edge-triggered with dedup + recovery -- and added 8 e2e tests in tests/alert_pipeline_dispatch.rs using a capturing AlertChannel sink)
- [ ] Test SLO tracking with synthetic uptime data (99.9% SLO with simulated downtime)
- [ ] Add test for capacity_planner with linear growth trend data verifying forecast accuracy
- [ ] Test metric_export Prometheus format output against promtool lint
- [ ] Add stress test: push 10K metrics/second and verify storage handles the load without data loss

## Documentation
- [ ] Document the feature flag matrix (sqlite, gpu, email, integrations) and which modules each enables
- [ ] Add deployment guide for running with external Prometheus/Grafana stack
- [ ] Document the alert rule expression syntax with examples for common conditions

## 0.1.8 Wave 5 (completed 2026-05-29)
- [x] Register 14 orphan modules via pub mod declarations (Slice ξ)
  - Modules: capacity_advisor, cardinality, dashboard_layout, gpu_metrics_sysfs (cfg linux), incident_manager,
    metric_downsample, metric_downsampling, opentelemetry_export, seasonal, sla_tracker, sqlite_pool (cfg feature),
    statsd_parser, trace_context, trace_exporter
  - File: src/lib.rs; dedup decisions documented inline
