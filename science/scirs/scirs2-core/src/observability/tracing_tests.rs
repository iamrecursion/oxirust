use super::*;

use super::*;
use std::thread;
use std::time::Duration;

#[test]
fn test_trace_context_creation() {
    let context = TraceContext::new();
    assert!(context.is_sampled());
    assert!(context.parent_spanid.is_none());

    let child = context.child();
    assert_eq!(child.trace_id, context.trace_id);
    assert_eq!(child.parent_spanid, Some(context.spanid));
    assert_ne!(child.spanid, context.spanid);
}

#[test]
fn test_tracingsystem_creation() {
    let config = TracingConfig::default();
    let tracing = TracingSystem::new(config).expect("Failed to create tracing system");

    let metrics = tracing.get_metrics().expect("Failed to get metrics");
    assert_eq!(metrics.spans_started, 0);
    assert_eq!(metrics.active_spans, 0);
}

#[test]
fn test_span_lifecycle() {
    let config = TracingConfig::default();
    let tracing = TracingSystem::new(config).expect("Failed to create tracing system");

    {
        let span = tracing
            .start_span("test_operation")
            .expect("Failed to start span");
        span.add_attribute("test", "value")
            .expect("Failed to add attribute");
        span.add_event("test_event", HashMap::new())
            .expect("Failed to add event");

        let context = span.context().expect("Failed to get context");
        assert!(context.is_sampled());
    } // Span ends here

    // Give some time for cleanup
    thread::sleep(Duration::from_millis(10));

    let metrics = tracing.get_metrics().expect("Failed to get metrics");
    assert_eq!(metrics.spans_started, 1);
}

#[test]
fn test_span_builder() {
    let config = TracingConfig::default();
    let tracing = TracingSystem::new(config).expect("Failed to create tracing system");

    let span = SpanBuilder::new("test_operation")
        .with_kind(SpanKind::Server)
        .with_attribute("method", "GET")
        .with_component("web_server")
        .start(&tracing)
        .expect("Failed to start span");

    let context = span.context().expect("Failed to get context");
    assert!(context.is_sampled());
}

#[test]
fn test_probability_sampler() {
    let sampler = ProbabilitySampler::new(0.0);
    let context = TraceContext::new();
    assert!(!sampler.should_sample(&context, "test"));

    let sampler = ProbabilitySampler::new(1.0);
    assert!(sampler.should_sample(&context, "test"));
}

#[test]
fn test_console_exporter() {
    let exporter = ConsoleExporter::new(false);
    let context = TraceContext::new();

    let span = Span {
        context,
        name: "test".to_string(),
        kind: SpanKind::Internal,
        start_time: SystemTime::now(),
        end_time: Some(SystemTime::now()),
        status: SpanStatus::Ok,
        attributes: HashMap::new(),
        events: Vec::new(),
        metrics: SpanMetrics::default(),
        component: None,
        error: None,
    };

    // This will print to console
    exporter.export_span(&span).expect("Failed to export span");
}

#[test]
fn test_nested_spans() {
    let config = TracingConfig::default();
    let tracing = TracingSystem::new(config).expect("Failed to create tracing system");

    let parent_span = tracing
        .start_span("parent_operation")
        .expect("Failed to start parent span");
    let parent_context = parent_span.context().expect("Failed to get parent context");

    let child_span = SpanBuilder::new("child_operation")
        .with_parent(parent_context.clone())
        .start(&tracing)
        .expect("Failed to start child span");

    let child_context = child_span.context().expect("Failed to get child context");
    assert_eq!(child_context.trace_id, parent_context.trace_id);
    assert_eq!(child_context.parent_spanid, Some(parent_context.spanid));
}

#[test]
fn test_w3c_trace_context() {
    let context = TraceContext::new();
    let traceparent = context.to_traceparent();

    // Traceparent should have format: version-trace_id-spanid-flags
    let parts: Vec<&str> = traceparent.split('-').collect();
    assert_eq!(parts.len(), 4);
    assert_eq!(parts[0], "00"); // version
    assert_eq!(parts[3], "01"); // sampled flag

    // Test parsing back
    let parsed_context =
        TraceContext::from_traceparent(&traceparent).expect("Failed to parse traceparent");
    assert_eq!(parsed_context.trace_id, context.trace_id);
    assert!(parsed_context.is_remote);
    assert!(parsed_context.is_sampled());
}

#[test]
fn test_adaptive_sampler() {
    let sampler = AdaptiveSampler::new(0.5, 100.0);
    let context = TraceContext::new();

    // Sample some spans
    for _ in 0..10 {
        sampler.should_sample(&context, "test");
    }

    let (total, sampled, rate) = sampler.get_stats();
    assert_eq!(total, 10);
    assert!((0.0..=1.0).contains(&rate));
}

#[test]
fn test_rate_limiting_sampler() {
    let sampler = RateLimitingSampler::new(5); // Max 5 samples per second
    let context = TraceContext::new();

    // Should accept first 5 samples
    for i in 0..5 {
        assert!(
            sampler.should_sample(&context, "test"),
            "Sample {i} should be accepted"
        );
    }

    // Should reject further samples in the same window
    assert!(!sampler.should_sample(&context, "test"));
    assert!(!sampler.should_sample(&context, "test"));
}

#[test]
fn test_batch_exporter() {
    let console_exporter = ConsoleExporter::new(false);
    let batch_exporter = BatchExporter::new(
        Box::new(console_exporter),
        3,                      // batch size
        Duration::from_secs(1), // timeout
    );

    let context = TraceContext::new();
    let span = Span {
        context,
        name: "test".to_string(),
        kind: SpanKind::Internal,
        start_time: SystemTime::now(),
        end_time: Some(SystemTime::now()),
        status: SpanStatus::Ok,
        attributes: HashMap::new(),
        events: Vec::new(),
        metrics: SpanMetrics::default(),
        component: None,
        error: None,
    };

    // Export spans - should batch until threshold
    batch_exporter
        .export_span(&span)
        .expect("Failed to export span");
    batch_exporter
        .export_span(&span)
        .expect("Failed to export span");
    batch_exporter
        .export_span(&span)
        .expect("Failed to export span"); // Should trigger flush

    batch_exporter.flush().expect("Failed to flush");
}

#[test]
fn test_resource_attribution() {
    let mut attribution = ResourceAttribution::new()
            .with_cpu_time(1_000_000) // 1ms
            .with_memory_allocation(1024) // 1KB
            .with_io_stats(5, 100, 200); // 5 ops, 100 read, 200 written

    let other = ResourceAttribution::new()
            .with_cpu_time(500_000) // 0.5ms
            .with_memory_allocation(512); // 0.5KB

    attribution.merge(&other);

    assert_eq!(attribution.cpu_timens, Some(1_500_000));
    assert_eq!(attribution.memory_allocated_bytes, Some(1536));
    assert_eq!(attribution.io_operations, Some(5));
}

#[test]
fn test_enhanced_span_metrics() {
    let mut metrics = EnhancedSpanMetrics::new();
    metrics.add_performance_counter("cache_hits", 150);
    metrics.add_performance_counter("cache_misses", 25);
    metrics.add_performance_counter("cache_hits", 50); // Should add to existing

    assert_eq!(metrics.performance_counters.get("cache_hits"), Some(&200));
    assert_eq!(metrics.performance_counters.get("cache_misses"), Some(&25));

    // Test resource cost calculation
    metrics.resources.cpu_timens = Some(1_000_000); // 1ms
    metrics.resources.memory_allocated_bytes = Some(1_048_576); // 1MB
    metrics.resources.io_operations = Some(5);

    let cost = metrics.get_total_resource_cost();
    assert!(cost > 0.0);
}

#[test]
fn test_tracing_version_compatibility() {
    let v1_0 = TracingVersion::new(1, 0, 0);
    let v1_1 = TracingVersion::new(1, 1, 0);
    let v2_0 = TracingVersion::new(2, 0, 0);

    assert!(v1_0.is_compatible(&v1_1));
    assert!(!v1_1.is_compatible(&v1_0)); // Newer minor not compatible with older
    assert!(!v1_0.is_compatible(&v2_0)); // Different major versions not compatible

    assert_eq!(v1_0.to_string(), "1.0.0");
}
