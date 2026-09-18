//! # Advanced Profiling and Instrumentation Demo for SciRS2 v0.2.0
//!
//! This example demonstrates the comprehensive profiling and instrumentation
//! capabilities added in v0.2.0, including:
//!
//! - Tracing framework with structured logging
//! - OpenTelemetry integration for distributed tracing
//! - Prometheus metrics export
//! - Instrumentation framework
//! - Platform-specific profiling (Linux perf)
//! - Advanced memory profiling
//!
//! Run with:
//! ```bash
//! cargo run --example advanced_profiling_demo --features "profiling_all"
//! ```

use scirs2_core::CoreResult;

#[cfg(feature = "profiling_advanced")]
use scirs2_core::profiling::tracing_framework::{init_tracing, PerfZone, SpanGuard, TracingConfig};
#[cfg(feature = "profiling_advanced")]
use tracing::{debug, info, warn, Level};

#[cfg(feature = "profiling_prometheus")]
use scirs2_core::profiling::prometheus_metrics::{
    latency_buckets, register_counter, register_histogram, MetricsRegistry, SciRS2Metrics,
};

#[cfg(feature = "instrumentation")]
use scirs2_core::profiling::instrumentation::{
    get_counter, print_counter_summary, record_event, InstrumentationScope,
};

#[cfg(all(target_os = "linux", feature = "profiling_perf"))]
use scirs2_core::profiling::perf_integration::{PerfCounter, PerfCounterGroup, PerfEvent};

#[cfg(feature = "profiling_memory")]
use scirs2_core::profiling::memory_profiling::{AllocationTracker, MemoryProfiler};

fn main() -> CoreResult<()> {
    println!("🚀 SciRS2 v0.2.0 Advanced Profiling and Instrumentation Demo");
    println!("===========================================================\n");

    // 1. Initialize tracing framework
    #[cfg(feature = "profiling_advanced")]
    {
        demo_tracing_framework()?;
    }

    // 2. Demonstrate Prometheus metrics
    #[cfg(feature = "profiling_prometheus")]
    {
        demo_prometheus_metrics()?;
    }

    // 3. Demonstrate instrumentation framework
    #[cfg(feature = "instrumentation")]
    {
        demo_instrumentation()?;
    }

    // 4. Demonstrate platform-specific profiling
    #[cfg(all(target_os = "linux", feature = "profiling_perf"))]
    {
        demo_perf_integration()?;
    }

    // 5. Demonstrate memory profiling
    #[cfg(feature = "profiling_memory")]
    {
        demo_memory_profiling()?;
    }

    println!("\n✨ Advanced profiling demo completed successfully!");
    println!("\n📊 Summary of v0.2.0 Profiling Features:");
    println!("  ✓ Structured logging with tracing framework");
    println!("  ✓ OpenTelemetry integration for distributed tracing");
    println!("  ✓ Prometheus metrics export");
    println!("  ✓ Instrumentation framework with zero overhead");
    println!("  ✓ Platform-specific profiling (Linux perf)");
    println!("  ✓ Advanced memory profiling with jemalloc");

    Ok(())
}

#[cfg(feature = "profiling_advanced")]
fn demo_tracing_framework() -> CoreResult<()> {
    println!("📝 1. Tracing Framework Demo");
    println!("=============================\n");

    // Initialize tracing with development configuration
    let config = TracingConfig::development();
    let _guard = init_tracing(config)?;

    info!("Tracing framework initialized");

    // Create a span for computation
    {
        let _span = SpanGuard::new("matrix_computation");
        info!(size = 1000, "Starting matrix computation");

        // Simulate computation with nested spans
        {
            let _nested = SpanGuard::new("matrix_multiply");
            debug!("Performing matrix multiplication");
            std::thread::sleep(std::time::Duration::from_millis(50));
        }

        {
            let _nested = SpanGuard::new("matrix_invert");
            debug!("Performing matrix inversion");
            std::thread::sleep(std::time::Duration::from_millis(30));
        }

        info!(result = "success", "Matrix computation completed");
    }

    // Use performance zones
    {
        let zone = PerfZone::start("critical_section");
        warn!("Entering critical section");
        std::thread::sleep(std::time::Duration::from_millis(20));
        zone.end();
    }

    println!("  ✓ Tracing framework demo completed\n");
    Ok(())
}

#[cfg(feature = "profiling_prometheus")]
fn demo_prometheus_metrics() -> CoreResult<()> {
    println!("📊 2. Prometheus Metrics Demo");
    println!("==============================\n");

    // Register standard SciRS2 metrics
    let metrics = SciRS2Metrics::register()?;

    println!("  Registered SciRS2 metrics:");
    println!("    - operations_total");
    println!("    - operation_duration_seconds");
    println!("    - active_operations");
    println!("    - memory_usage_bytes");
    println!("    - array_size_bytes");
    println!("    - errors_total\n");

    // Record some metrics
    metrics
        .operations_total
        .with_label_values(&["matrix_multiply", "linalg"])
        .inc();

    metrics
        .operation_duration
        .with_label_values(&["matrix_multiply", "linalg"])
        .observe(0.123);

    metrics.memory_usage_bytes.set(1_048_576.0);
    metrics.array_size_bytes.observe(8192.0);

    // Register custom counter
    let custom_counter = register_counter(
        "scirs2_custom_operations",
        "Custom operation counter for demo",
    )?;
    custom_counter.inc();
    custom_counter.inc();

    // Register custom histogram
    let custom_histogram = register_histogram(
        "scirs2_custom_latency",
        "Custom latency histogram for demo",
        latency_buckets(),
    )?;
    custom_histogram.observe(0.045);
    custom_histogram.observe(0.123);
    custom_histogram.observe(0.089);

    // Gather and print metrics
    println!("  📈 Metrics Export (Prometheus format):");
    println!("  {}", "-".repeat(50));
    let metrics_text = MetricsRegistry::gather();
    // Print first few lines
    for line in metrics_text.lines().take(20) {
        println!("  {}", line);
    }
    println!("  {} ... (truncated)", "-".repeat(50));

    println!("\n  ✓ Prometheus metrics demo completed\n");
    Ok(())
}

#[cfg(feature = "instrumentation")]
fn demo_instrumentation() -> CoreResult<()> {
    println!("🔍 3. Instrumentation Framework Demo");
    println!("=====================================\n");

    // Record events
    record_event("demo_started", &[("version", &"0.2.0")]);

    // Use instrumentation scope
    {
        let _scope = InstrumentationScope::new("data_processing");
        std::thread::sleep(std::time::Duration::from_millis(30));

        record_event("data_processed", &[("records", &1000), ("time_ms", &30)]);
    }

    // Use performance counters
    let fft_counter = get_counter("fft_operations");
    for _ in 0..5 {
        let start = std::time::Instant::now();
        std::thread::sleep(std::time::Duration::from_millis(10));
        fft_counter.add_duration(start.elapsed());
    }

    let linalg_counter = get_counter("linalg_operations");
    for _ in 0..3 {
        let start = std::time::Instant::now();
        std::thread::sleep(std::time::Duration::from_millis(15));
        linalg_counter.add_duration(start.elapsed());
    }

    // Print counter summary
    println!("  📊 Performance Counter Summary:");
    println!("  {}", "-".repeat(50));
    print_counter_summary();
    println!("  {}", "-".repeat(50));

    println!("\n  ✓ Instrumentation framework demo completed\n");
    Ok(())
}

#[cfg(all(target_os = "linux", feature = "profiling_perf"))]
fn demo_perf_integration() -> CoreResult<()> {
    println!("⚡ 4. Linux Perf Integration Demo");
    println!("==================================\n");

    println!("  Note: This requires appropriate permissions (CAP_SYS_ADMIN)");
    println!("  If you see errors, try running with sudo or adjusting perf_event_paranoid\n");

    // Try to create performance counters
    let result = PerfCounterGroup::new(&[
        PerfEvent::CpuCycles,
        PerfEvent::Instructions,
        PerfEvent::CacheMisses,
    ]);

    match result {
        Ok(mut group) => {
            println!("  ✓ Performance counters created successfully");

            if group.enable().is_ok() {
                println!("  ✓ Performance counters enabled");

                // Perform some computation
                let mut sum = 0u64;
                for i in 0..1_000_000 {
                    sum = sum.wrapping_add(i);
                }

                if let Ok(results) = group.read() {
                    println!("\n  📊 Performance Counter Results:");
                    println!("  {}", "-".repeat(50));
                    for (event, value) in results {
                        println!("    {:?}: {}", event, value);
                    }
                    println!("  {}", "-".repeat(50));
                }

                let _ = group.disable();
            } else {
                println!("  ⚠️  Failed to enable performance counters (permission issue)");
            }
        }
        Err(e) => {
            println!("  ⚠️  Failed to create performance counters: {}", e);
            println!("  This is expected if not running with appropriate permissions");
        }
    }

    println!("\n  ✓ Perf integration demo completed\n");
    Ok(())
}

#[cfg(feature = "profiling_memory")]
fn demo_memory_profiling() -> CoreResult<()> {
    println!("💾 5. Memory Profiling Demo");
    println!("===========================\n");

    // Create memory profiler
    let mut profiler = MemoryProfiler::new();
    profiler.set_baseline()?;

    println!("  📊 Initial Memory Stats:");
    MemoryProfiler::print_stats()?;

    // Track allocations
    let mut tracker = AllocationTracker::new();
    tracker.snapshot("baseline")?;

    // Allocate some memory
    {
        let _vec1: Vec<u8> = vec![0; 1_000_000];
        tracker.snapshot("after_1mb_alloc")?;

        let _vec2: Vec<u8> = vec![0; 5_000_000];
        tracker.snapshot("after_5mb_alloc")?;

        let _vec3: Vec<u8> = vec![0; 10_000_000];
        tracker.snapshot("after_10mb_alloc")?;
    }

    tracker.snapshot("after_dealloc")?;

    // Analyze allocation patterns
    let analysis = tracker.analyze();
    println!("\n  📈 Allocation Analysis:");
    println!("  {}", "-".repeat(50));
    println!("    Total Allocated: {} bytes", analysis.total_allocated);
    println!("    Peak Allocated:  {} bytes", analysis.peak_allocated);
    println!("    Total Snapshots: {}", analysis.total_snapshots);

    if let Some((label, increase)) = analysis.largest_increase {
        println!("    Largest Increase: {} bytes (at {})", increase, label);
    }
    println!("  {}", "-".repeat(50));

    // Get memory delta
    if let Some(delta) = profiler.get_delta()? {
        println!("\n  📊 Memory Delta from Baseline:");
        println!("  {}", "-".repeat(50));
        println!("{}", delta.format());
        println!("  {}", "-".repeat(50));
    }

    println!("\n  ✓ Memory profiling demo completed\n");
    Ok(())
}
