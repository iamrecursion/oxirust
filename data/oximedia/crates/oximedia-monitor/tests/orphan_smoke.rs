//! Smoke tests verifying that all newly-wired orphan modules compile and can
//! instantiate their primary public types.
//!
//! These tests do **not** require the `sqlite` feature and run on all targets
//! except `wasm32`.

// ---------------------------------------------------------------------------
// capacity_advisor
// ---------------------------------------------------------------------------

#[test]
fn test_capacity_advisor_instantiation() {
    use oximedia_monitor::capacity_advisor::CapacityAdvisor;
    let _advisor = CapacityAdvisor::new();
}

#[test]
fn test_capacity_advisor_with_defaults() {
    use oximedia_monitor::capacity_advisor::CapacityAdvisor;
    let advisor = CapacityAdvisor::with_defaults();
    // Composite score should be valid on zero utilisation.
    let score = advisor.composite_score();
    assert!(score.overall >= 0.0 && score.overall <= 100.0);
}

// ---------------------------------------------------------------------------
// cardinality
// ---------------------------------------------------------------------------

#[test]
fn test_cardinality_guard_instantiation() {
    use oximedia_monitor::cardinality::{CardinalityConfig, CardinalityGuard};
    let cfg = CardinalityConfig::default_safe();
    let _guard = CardinalityGuard::new(cfg);
}

// ---------------------------------------------------------------------------
// dashboard_layout
// ---------------------------------------------------------------------------

#[test]
fn test_dashboard_layout_template_library() {
    use oximedia_monitor::dashboard_layout::TemplateLibrary;
    let tpl = TemplateLibrary::media_server_overview();
    assert!(tpl.validate().is_empty(), "Built-in template must be valid");
}

// ---------------------------------------------------------------------------
// incident_manager
// ---------------------------------------------------------------------------

#[test]
fn test_incident_manager_instantiation() {
    use oximedia_monitor::incident_manager::IncidentManager;
    let mgr = IncidentManager::new();
    // Should have zero incidents on creation.
    assert_eq!(mgr.total_count(), 0);
}

// ---------------------------------------------------------------------------
// metric_downsample
// ---------------------------------------------------------------------------

#[test]
fn test_metric_downsample_retention_policy() {
    use oximedia_monitor::metric_downsample::RetentionPolicy;
    let policy = RetentionPolicy::one_second_for_one_hour();
    assert!(policy.resolution_ms > 0);
    assert!(policy.retention_ms > 0);
}

#[test]
fn test_metric_tier_storage_instantiation() {
    use oximedia_monitor::metric_downsample::{MetricTierStorage, RetentionPolicy};
    let tiers = vec![
        RetentionPolicy::one_second_for_one_hour(),
        RetentionPolicy::one_minute_for_one_day(),
    ];
    let _storage = MetricTierStorage::new(tiers);
}

// ---------------------------------------------------------------------------
// metric_downsampling (LTTB etc.)
// ---------------------------------------------------------------------------

#[test]
fn test_metric_downsampling_lttb() {
    use oximedia_monitor::metric_downsampling::{DownsampleMethod, MetricDownsampler, Sample};

    let samples: Vec<Sample> = (0..200)
        .map(|i| Sample {
            timestamp_secs: i as u64,
            value: f64::from(i as i32 % 20),
        })
        .collect();

    let result = MetricDownsampler::downsample(&samples, 50, DownsampleMethod::Lttb)
        .expect("LTTB downsample should succeed");
    assert_eq!(result.original_count, 200);
    assert_eq!(result.samples.len(), 50);
}

#[test]
fn test_metric_downsampling_average() {
    use oximedia_monitor::metric_downsampling::{DownsampleMethod, MetricDownsampler, Sample};

    let samples: Vec<Sample> = (0..100)
        .map(|i| Sample {
            timestamp_secs: i as u64,
            value: 1.0,
        })
        .collect();

    let result = MetricDownsampler::downsample(&samples, 10, DownsampleMethod::Average)
        .expect("Average downsample should succeed");
    assert_eq!(result.samples.len(), 10);
}

// ---------------------------------------------------------------------------
// opentelemetry_export
// ---------------------------------------------------------------------------

#[test]
fn test_metric_bridge_convert() {
    use oximedia_monitor::opentelemetry_export::{InternalMetric, MetricBridge, OtelExportConfig};

    let config = OtelExportConfig::new("test-service", "0.1.8");
    let bridge = MetricBridge::new(&config);

    let internal = InternalMetric {
        name: "cpu_usage".to_string(),
        value: 72.5,
        labels: vec![("host".to_string(), "node-01".to_string())],
        timestamp_secs: 1_700_000_000,
    };

    let point = bridge.convert(&internal);
    assert!((point.value - 72.5_f64).abs() < f64::EPSILON);
}

// ---------------------------------------------------------------------------
// seasonal
// ---------------------------------------------------------------------------

#[test]
fn test_seasonal_decomposition_basic() {
    use oximedia_monitor::seasonal::SeasonalDecomposition;

    // 48 hourly samples = 2 daily cycles (period=24)
    let data: Vec<f32> = (0..48)
        .map(|i| {
            let base = 50.0f32;
            let seasonal = 10.0 * (i as f32 * std::f32::consts::TAU / 24.0).sin();
            base + seasonal
        })
        .collect();

    let decomp = SeasonalDecomposition::decompose(&data, 24)
        .map_err(|e| format!("decompose failed: {e}"))
        .expect("Decomposition should succeed for 48 samples with period 24");

    assert_eq!(decomp.original.len(), 48);
    assert_eq!(decomp.trend.len(), 48);
    assert_eq!(decomp.seasonal.len(), 24);
    assert_eq!(decomp.residual.len(), 48);
}

// ---------------------------------------------------------------------------
// sla_tracker
// ---------------------------------------------------------------------------

#[test]
fn test_sla_tracker_instantiation() {
    use oximedia_monitor::sla_tracker::{SlaTier, SlaTracker};

    let mut tracker = SlaTracker::new();
    // Window: start=0.0, end=86_400.0 (one day)
    tracker.register_service("media-encoder", SlaTier::Gold, 0.0, 86_400.0);
    assert_eq!(tracker.service_count(), 1);
}

// ---------------------------------------------------------------------------
// statsd_parser
// ---------------------------------------------------------------------------

#[test]
fn test_statsd_parser_gauge() {
    use oximedia_monitor::statsd_parser::{parse_statsd_line, StatsdMetricType};

    let m = parse_statsd_line("cpu.usage:42.5|g").expect("should parse gauge");
    assert_eq!(m.name, "cpu.usage");
    assert!((m.value - 42.5_f64).abs() < f64::EPSILON);
    assert_eq!(m.metric_type, StatsdMetricType::Gauge);
}

#[test]
fn test_statsd_parser_counter() {
    use oximedia_monitor::statsd_parser::{parse_statsd_line, StatsdMetricType};

    let m = parse_statsd_line("requests:1|c").expect("should parse counter");
    assert_eq!(m.metric_type, StatsdMetricType::Counter);
    assert!((m.value - 1.0_f64).abs() < f64::EPSILON);
}

#[test]
fn test_statsd_parser_timer() {
    use oximedia_monitor::statsd_parser::{parse_statsd_line, StatsdMetricType};

    let m = parse_statsd_line("response_time:350|ms").expect("should parse timer");
    assert_eq!(m.metric_type, StatsdMetricType::Timer);
    assert!((m.value - 350.0_f64).abs() < f64::EPSILON);
}

// ---------------------------------------------------------------------------
// trace_context
// ---------------------------------------------------------------------------

#[test]
fn test_trace_context_generate_and_parse() {
    use oximedia_monitor::trace_context::TraceParent;

    let tp = TraceParent::new(42);
    let header = tp.to_header();
    let parsed = TraceParent::parse(&header).expect("should parse back");
    assert_eq!(tp.trace_id, parsed.trace_id);
    assert_eq!(tp.parent_id, parsed.parent_id);
    assert_eq!(tp.flags, parsed.flags);
}

// ---------------------------------------------------------------------------
// trace_exporter
// ---------------------------------------------------------------------------

#[test]
fn test_trace_exporter_instantiation() {
    use oximedia_monitor::trace_exporter::{ExporterConfig, TraceExporter};

    let config = ExporterConfig::default().with_service_name("smoke-test");
    let _exporter = TraceExporter::with_logging_sink(config);
}

// ---------------------------------------------------------------------------
// audio (broadcast metering)
// ---------------------------------------------------------------------------

#[test]
fn test_audio_meter_creation_and_process() {
    use oximedia_monitor::audio::AudioMeter;

    let mut meter = AudioMeter::new(48_000.0, 2).expect("should create 48 kHz stereo meter");
    let samples = vec![0.1f32; 4800]; // 100ms of stereo
    assert!(
        meter.process_samples(&samples).is_ok(),
        "should process samples without error"
    );
}

// ---------------------------------------------------------------------------
// cc (closed caption)
// ---------------------------------------------------------------------------

#[test]
fn test_caption_monitor_instantiation() {
    use oximedia_monitor::cc::CaptionMonitor;
    let _monitor = CaptionMonitor::new().expect("CaptionMonitor should init");
}

// ---------------------------------------------------------------------------
// compliance
// ---------------------------------------------------------------------------

#[test]
fn test_compliance_checker_ebu() {
    use oximedia_monitor::compliance::{ComplianceChecker, ComplianceStandard};

    let result = ComplianceChecker::new(ComplianceStandard::EbuR128);
    assert!(result.is_ok(), "EBU R128 compliance checker should init");
}

#[test]
fn test_compliance_checker_smpte() {
    use oximedia_monitor::compliance::{ComplianceChecker, ComplianceStandard};

    let result = ComplianceChecker::new(ComplianceStandard::Smpte);
    assert!(result.is_ok(), "SMPTE compliance checker should init");
}

// ---------------------------------------------------------------------------
// multiviewer
// ---------------------------------------------------------------------------

#[test]
fn test_multiviewer_instantiation() {
    use oximedia_monitor::multiviewer::{MultiViewer, ViewerLayout};
    let _viewer = MultiViewer::new(ViewerLayout::Grid2x2);
}

// ---------------------------------------------------------------------------
// reference
// ---------------------------------------------------------------------------

#[test]
fn test_reference_comparator_instantiation() {
    use oximedia_monitor::reference::ReferenceComparator;
    let _cmp = ReferenceComparator::new().expect("ReferenceComparator should init");
}

// ---------------------------------------------------------------------------
// scopes (video scope monitoring)
// ---------------------------------------------------------------------------

#[test]
fn test_scope_monitor_creation() {
    use oximedia_monitor::scopes::ScopeMonitor;
    use oximedia_scopes::ScopeConfig;

    let result = ScopeMonitor::new(ScopeConfig::default());
    assert!(
        result.is_ok(),
        "ScopeMonitor should create with default config"
    );
}

// ---------------------------------------------------------------------------
// status
// ---------------------------------------------------------------------------

#[test]
fn test_signal_monitor_instantiation() {
    use oximedia_monitor::status::SignalMonitor;
    let _mon = SignalMonitor::new();
}

// ---------------------------------------------------------------------------
// technical
// ---------------------------------------------------------------------------

#[test]
fn test_technical_analyzer_instantiation() {
    use oximedia_monitor::technical::TechnicalAnalyzer;
    let _result = TechnicalAnalyzer::new().expect("TechnicalAnalyzer should init");
}

// ---------------------------------------------------------------------------
// timecode
// ---------------------------------------------------------------------------

#[test]
fn test_timecode_monitor_creation() {
    use oximedia_monitor::timecode::TimecodeMonitor;
    use oximedia_timecode::FrameRate;

    let result = TimecodeMonitor::new(FrameRate::Fps25);
    assert!(result.is_ok(), "TimecodeMonitor should create for 25fps");
}

// ---------------------------------------------------------------------------
// focus (peaking)
// ---------------------------------------------------------------------------

#[test]
fn test_focus_peaking_instantiation() {
    use oximedia_monitor::focus::FocusPeaking;
    let _focus = FocusPeaking::new();
}

// ---------------------------------------------------------------------------
// gpu_metrics_sysfs (non-wasm32 only)
// ---------------------------------------------------------------------------

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn test_gpu_metrics_collector_instantiation() {
    use oximedia_monitor::gpu_metrics_sysfs::{GpuMetricsCollector, GpuVendor};
    let collector = GpuMetricsCollector::new();
    // collect() may return empty on CI machines without GPUs — that is fine.
    let _snapshots = collector.collect();
    // Verify the enum discriminants are available.
    let _vendor = GpuVendor::Amd;
}

// ---------------------------------------------------------------------------
// sqlite_pool (sqlite feature only)
// ---------------------------------------------------------------------------

#[cfg(all(not(target_arch = "wasm32"), feature = "sqlite"))]
#[test]
fn test_sqlite_pool_open_and_acquire() {
    use oximedia_monitor::sqlite_pool::{PoolConfig, SqlitePool};
    use std::time::Duration;

    let dir = std::env::temp_dir().join(format!(
        "oximedia_monitor_smoke_{}.db",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0)
    ));

    let config = PoolConfig::builder()
        .db_path(&dir)
        .pool_size(2)
        .acquire_timeout(Duration::from_secs(2))
        .build();

    let pool = SqlitePool::open(config).expect("pool should open");
    pool.with_connection(|conn| {
        conn.execute_batch("CREATE TABLE IF NOT EXISTS _smoke (v INTEGER)")
            .map_err(|e| oximedia_monitor::MonitorError::Storage(e.to_string()))
    })
    .expect("schema init should succeed");

    let _ = std::fs::remove_file(&dir);
}
