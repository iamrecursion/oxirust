//! Smoke tests for newly-wired orphan modules in `oximedia-profiler`.

// ── session ───────────────────────────────────────────────────────────────────
#[test]
fn profiling_session_record_and_finalise() {
    use oximedia_profiler::session::{ProfilingSession, SessionMetric};
    let mut session = ProfilingSession::new("test-session");
    session.record_metric("frames", SessionMetric::Count(1000));
    session.record_metric("fps", SessionMetric::Rate(59.94));
    session.finalise();
    assert_eq!(session.name(), "test-session");
    assert_eq!(session.event_count(), 0);
    let json = session.to_json();
    assert!(json.is_ok());
}

// ── event_ring_buffer ─────────────────────────────────────────────────────────
#[test]
fn event_ring_buffer_capacity_and_overflow() {
    use oximedia_profiler::event_ring_buffer::EventRingBuffer;
    let mut rb: EventRingBuffer<u64, 4> = EventRingBuffer::new();
    assert!(rb.is_empty());
    assert_eq!(rb.capacity(), 4);

    rb.push(10);
    rb.push(20);
    rb.push(30);
    rb.push(40);
    assert!(rb.is_full());
    assert_eq!(rb.len(), 4);

    // Overwrite oldest entry (10) with 50
    rb.push(50);
    assert_eq!(rb.len(), 4); // still full
                             // Last pushed should be accessible
    assert_eq!(rb.last(), Some(50));
}

#[test]
fn event_ring_buffer_clear() {
    use oximedia_profiler::event_ring_buffer::EventRingBuffer;
    let mut rb: EventRingBuffer<u32, 8> = EventRingBuffer::new();
    rb.push(1);
    rb.push(2);
    rb.clear();
    assert!(rb.is_empty());
}

// ── hierarchical_span ─────────────────────────────────────────────────────────
#[test]
fn span_report_from_empty_trees() {
    use oximedia_profiler::hierarchical_span::SpanReport;
    // SpanReport::new always includes a 2-line header even for empty input.
    let report = SpanReport::new(&[]);
    // No tree entries → only the 2 header lines.
    assert_eq!(report.line_count(), 2);
    let formatted = report.format();
    assert!(formatted.contains("Hierarchical Span Report"));
}

// ── chrome_tracing ────────────────────────────────────────────────────────────
#[test]
fn chrome_tracing_exporter_add_events() {
    use oximedia_profiler::chrome_tracing::ChromeTracingExporter;
    let mut exp = ChromeTracingExporter::new();
    exp.add_begin("decode", "codec", 0.0, 1, 1);
    exp.add_end("decode", "codec", 500.0, 1, 1);
    assert_eq!(exp.event_count(), 2);
}

#[test]
fn trace_event_phase_detection() {
    use oximedia_profiler::chrome_tracing::TraceEvent;
    let begin = TraceEvent::begin("render", "gpu", 0.0, 1, 1);
    let complete = TraceEvent::complete("process", "cpu", 100.0, 200.0, 1, 1);
    let instant = TraceEvent::instant("checkpoint", "misc", 50.0, 1, 1);
    assert!(!begin.is_complete());
    assert!(!begin.is_instant());
    assert!(complete.is_complete());
    assert!(instant.is_instant());
}

// ── flame_graph ───────────────────────────────────────────────────────────────
#[test]
fn flame_graph_builder_single_stack() {
    use oximedia_profiler::flame_graph::FlameGraphBuilder;
    let mut builder = FlameGraphBuilder::new();
    builder
        .record(&["main", "encode", "write"], 1000)
        .expect("record should succeed");
    let graph = builder.build().expect("build should succeed");
    assert!(graph.total_nodes() > 0);
    let folded = graph.to_folded();
    assert!(folded.contains("main"));
}

// ── frame_budget ──────────────────────────────────────────────────────────────
#[test]
fn frame_budget_tracker_60fps_config() {
    use oximedia_profiler::frame_budget::FrameBudgetTracker;
    let tracker = FrameBudgetTracker::at_60fps();
    assert!(!tracker.is_frame_open());
    assert!(tracker.frames().is_empty());
    // 60fps budget ≈ 16.67ms
    let budget_ms = tracker.config().budget_ms;
    assert!((budget_ms - 16.67).abs() < 0.5, "budget_ms={budget_ms}");
}

// ── distributed_profile ───────────────────────────────────────────────────────
#[test]
fn distributed_profile_aggregator_empty() {
    use oximedia_profiler::distributed_profile::DistributedProfileAggregator;
    let agg = DistributedProfileAggregator::new();
    let profile = agg.aggregate();
    assert_eq!(profile.total_spans(), 0);
    assert_eq!(profile.node_count(), 0);
}

// ── pipeline_bottleneck ───────────────────────────────────────────────────────
#[test]
fn pipeline_analyzer_no_stages() {
    use oximedia_profiler::pipeline_bottleneck::PipelineAnalyzer;
    let analyzer = PipelineAnalyzer::new();
    assert!(analyzer.stage_timings().is_empty());
}

// ── power_energy ──────────────────────────────────────────────────────────────
#[test]
fn power_domain_labels_defined() {
    use oximedia_profiler::power_energy::PowerDomain;
    // Verify label() doesn't panic for any variant.
    let domains = [
        PowerDomain::Package,
        PowerDomain::Cores,
        PowerDomain::Gpu,
        PowerDomain::Dram,
        PowerDomain::Platform,
    ];
    for domain in domains {
        assert!(!domain.label().is_empty());
    }
}

// ── profile_compare ───────────────────────────────────────────────────────────
#[test]
fn profile_comparator_no_regressions_identical_snapshots() {
    use oximedia_profiler::profile_compare::{ProfileComparator, ProfileSnapshot};
    use std::time::Duration;
    let mut baseline = ProfileSnapshot::new();
    let durations = [
        Duration::from_millis(100),
        Duration::from_millis(110),
        Duration::from_millis(105),
    ];
    baseline.record_durations("encode", &durations);
    let comparator = ProfileComparator::default();
    let report = comparator.compare(&baseline, &baseline);
    assert!(
        !report.has_regressions(),
        "identical snapshots should have no regressions"
    );
}

// ── tls_counters ──────────────────────────────────────────────────────────────
#[test]
fn tls_counter_registry_empty() {
    use oximedia_profiler::tls_counters::TlsCounterRegistry;
    let registry = TlsCounterRegistry::new();
    let agg = registry.aggregate();
    // Fresh registry produces empty counters (no counts).
    assert_eq!(agg.contributing_threads, 0);
    assert!(agg.counts.is_empty());
}

// ── mem_stage_profiler ────────────────────────────────────────────────────────
#[test]
fn memory_profiler_empty_total_profile() {
    use oximedia_profiler::mem_stage_profiler::MemoryProfiler;
    let profiler = MemoryProfiler::new();
    let profile = profiler.total_profile();
    assert_eq!(profile.allocation_count, 0);
    assert_eq!(profile.total_allocated, 0);
    assert_eq!(profile.total_freed, 0);
}

// ── perf_script ───────────────────────────────────────────────────────────────
#[test]
fn perf_script_export_empty_output() {
    use oximedia_profiler::perf_script::PerfScriptExporter;
    let output = PerfScriptExporter::export(&[]);
    assert!(output.is_empty(), "empty export should produce no output");
}

// ── session (JSON roundtrip) ──────────────────────────────────────────────────
#[test]
fn profiling_session_json_roundtrip() {
    use oximedia_profiler::session::{ProfilingSession, SessionMetric};
    let mut session = ProfilingSession::new("roundtrip-test");
    session.record_metric(
        "latency_ms",
        SessionMetric::Duration(std::time::Duration::from_millis(42)),
    );
    session.finalise();
    let json = session.to_json().expect("to_json should succeed");
    let restored = ProfilingSession::from_json(&json).expect("from_json should succeed");
    assert_eq!(restored.name(), "roundtrip-test");
}
