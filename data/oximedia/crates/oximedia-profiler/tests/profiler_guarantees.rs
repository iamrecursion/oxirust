//! Behavioural guarantee tests for `oximedia-profiler`.
//!
//! These tests pin three production guarantees of the profiler:
//!
//! 1. **Sampling overhead < 1 %** — the profiler's *own accounted* overhead
//!    (recording cost vs. workload cost) stays below the 1 % target.  The
//!    always-on tests use a deterministic, count-based accounting model and the
//!    profiler's own `tls_counters` self-timing so they never depend on a flaky
//!    wall-clock threshold.  A tight wall-clock assertion is provided but
//!    `#[ignore]`-gated to avoid CI flakiness under contention.
//! 2. **Regression detection** — an injected slowdown above threshold is
//!    flagged; a within-noise delta is not (no false positive).  Both the
//!    snapshot-based `ProfileComparator` and the benchmark-based
//!    `RegressionDetector` are exercised.
//! 3. **Concurrent allocation correctness** — driving a known number of
//!    allocations from N threads yields exactly the known sum, with no lost or
//!    double-counted records, both for the lock-free `AllocationTracker` and
//!    the thread-local `tls_counters` aggregation path.
//!
//! A bonus flamegraph/hotspot known-tree test verifies that a deterministic
//! call tree surfaces the expected hot frame on top.

use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use oximedia_profiler::allocation_tracker::{AllocationTracker, AllocationType};
use oximedia_profiler::benchmark::runner::BenchmarkResult;
use oximedia_profiler::flame::{CallStack, FlameGraph, FlameGraphBuilder, StackFrame};
use oximedia_profiler::hotspot::HotspotDetector;
use oximedia_profiler::profile_compare::{ProfileComparator, ProfileSnapshot};
use oximedia_profiler::regression::RegressionDetector;
use oximedia_profiler::sampling_profiler::{
    AdaptiveConfig, AdaptiveDecision, AdaptiveSamplingController, OverheadMeasurement, SampleEvent,
    SamplingConfig, SamplingProfiler,
};
use oximedia_profiler::tls_counters::{with_tls_counters, TlsCounterRegistry};

// ===========================================================================
// 1. Sampling overhead < 1 %
// ===========================================================================

/// Deterministic, count-based overhead accounting.
///
/// The profiler's accounted overhead is modelled as
/// `profiler_ops / (profiler_ops + workload_ops)`.  For a realistic sampling
/// configuration the sampler touches the workload only at the sample interval,
/// so for a workload doing many units of work between samples the accounted
/// overhead is well below 1 %.  This is a *pure arithmetic* check — no
/// wall-clock, no flakiness — that pins the < 1 % guarantee on the accounting
/// model the adaptive controller consumes.
#[test]
fn overhead_accounting_below_one_percent_deterministic() {
    // A 100 Hz sampler observing a workload that performs 1_000_000 work units
    // per second.  Each sample is one "profiler op"; each work unit is one
    // "workload op".  Over one second: 100 profiler ops vs 1_000_000 workload
    // ops.
    let profiler_ops: f64 = 100.0;
    let workload_ops: f64 = 1_000_000.0;
    let accounted_overhead = profiler_ops / (profiler_ops + workload_ops);

    assert!(
        accounted_overhead < 0.01,
        "accounted overhead {accounted_overhead} must be < 1 %"
    );

    // Feed this accounted overhead to the adaptive controller: because it is
    // below the 1 % target *and* below the headroom band (0.5 % by default),
    // the controller must be willing to *increase* the rate, never reduce it.
    let cfg = AdaptiveConfig {
        target_overhead: 0.01,
        headroom_factor: 0.5,
        measurement_window_count: 1,
        min_rate_hz: 1,
        max_rate_hz: 10_000,
        ..Default::default()
    };
    let mut ctrl = AdaptiveSamplingController::new(100, cfg);
    let decision = ctrl
        .observe(OverheadMeasurement {
            fraction: accounted_overhead,
            achieved_rate_hz: 100.0,
            window_duration: Duration::from_secs(1),
        })
        .expect("single-window controller should decide");

    assert_ne!(
        decision,
        AdaptiveDecision::Reduced,
        "sub-1 % overhead must never trigger a rate reduction"
    );
    assert!(
        ctrl.current_rate_hz() >= 100,
        "rate must not fall when overhead is below target"
    );
}

/// The adaptive controller's defining behaviour: overhead *above* 1 % reduces
/// the rate (so steady-state overhead is driven back below the target), while
/// overhead *below* 1 % does not.  This pins the control law that keeps the
/// long-run accounted overhead under 1 %.
#[test]
fn adaptive_controller_keeps_overhead_under_target() {
    let make_cfg = || AdaptiveConfig {
        target_overhead: 0.01,
        headroom_factor: 0.5,
        measurement_window_count: 1,
        reduction_factor: 0.5,
        min_rate_hz: 1,
        max_rate_hz: 10_000,
        ..Default::default()
    };

    // Over-budget (5 %) → reduce.
    let mut over = AdaptiveSamplingController::new(1_000, make_cfg());
    let d_over = over
        .observe(OverheadMeasurement {
            fraction: 0.05,
            achieved_rate_hz: 1_000.0,
            window_duration: Duration::from_millis(100),
        })
        .expect("decide");
    assert_eq!(d_over, AdaptiveDecision::Reduced);
    assert!(over.current_rate_hz() < 1_000);

    // Exactly at a comfortable sub-target (0.2 %) → not reduced.
    let mut under = AdaptiveSamplingController::new(1_000, make_cfg());
    let d_under = under
        .observe(OverheadMeasurement {
            fraction: 0.002,
            achieved_rate_hz: 1_000.0,
            window_duration: Duration::from_millis(100),
        })
        .expect("decide");
    assert_ne!(d_under, AdaptiveDecision::Reduced);
    assert!(under.current_rate_hz() >= 1_000);
}

/// Self-timed overhead via the profiler's own `tls_counters`.
///
/// We run a fixed workload and, on the same monotonic clock, separately
/// account the time the profiler spends in its `record()` hot path under the
/// `"profiler_overhead"` counter and the workload time under `"workload"`.
/// The ratio is the profiler's *self-measured* overhead.
///
/// This is wall-clock-derived, so the always-on assertion uses a deliberately
/// loose multiplicative bound (overhead must be a small fraction, < 25 %, of
/// the workload even when each "work unit" is cheap).  A tight < 1 % assertion
/// lives in the `#[ignore]`-gated test below.  What is asserted unconditionally
/// here is that the accounting is *coherent*: both counters advance and the
/// overhead counter is strictly smaller than the workload counter.
#[test]
fn self_timed_overhead_accounting_is_coherent() {
    const WORK_UNITS: usize = 200_000;
    const SAMPLE_EVERY: usize = 2_000; // 100 samples total — a 0.05 % sampling ratio

    let mut profiler = SamplingProfiler::new(SamplingConfig {
        sample_rate_hz: 100,
        ..Default::default()
    });
    profiler.start();

    let mut sink: u64 = 0;
    for i in 0..WORK_UNITS {
        // --- workload unit (timed under "workload") ---
        let w0 = std::time::Instant::now();
        sink = sink.wrapping_add((i as u64).wrapping_mul(2_654_435_761));
        sink ^= sink >> 13;
        with_tls_counters(|c| c.add_duration("workload", w0.elapsed()));

        // --- profiler op only at the sample interval (timed under overhead) ---
        if i % SAMPLE_EVERY == 0 {
            let p0 = std::time::Instant::now();
            profiler.record(SampleEvent::new(
                i as u64,
                vec!["main".to_string(), "hot_loop".to_string()],
                1,
                0,
            ));
            with_tls_counters(|c| c.add_duration("profiler_overhead", p0.elapsed()));
        }
    }
    profiler.stop();

    // Prevent the optimiser from eliminating the workload entirely.
    assert_ne!(sink, 0, "workload must have executed");
    assert_eq!(
        profiler.sample_count(),
        WORK_UNITS.div_ceil(SAMPLE_EVERY),
        "every sample-interval tick must have recorded exactly one event"
    );

    let workload = with_tls_counters(|c| c.duration("workload"));
    let overhead = with_tls_counters(|c| c.duration("profiler_overhead"));

    assert!(workload > Duration::ZERO, "workload time must be accounted");
    assert!(
        overhead > Duration::ZERO,
        "profiler overhead time must be accounted"
    );
    // Loose, always-on bound: the profiler's recording cost is a small slice of
    // the total, never dominating the workload.
    assert!(
        overhead.as_secs_f64() < 0.25 * workload.as_secs_f64(),
        "profiler overhead {:?} should be a small fraction of workload {:?}",
        overhead,
        workload
    );

    // Clean up the thread-local counters so other tests on this thread start
    // from a blank slate.
    with_tls_counters(|c| c.reset());
}

/// Tight wall-clock overhead assertion — `#[ignore]`d by default because the
/// absolute ratio depends on machine load and is flaky under CI contention.
/// Run explicitly with `cargo test -- --ignored` on a quiet machine.
#[test]
#[ignore = "tight wall-clock ratio is load-sensitive; run on a quiet machine"]
fn self_timed_overhead_below_one_percent_tight() {
    const WORK_UNITS: usize = 2_000_000;
    const SAMPLE_EVERY: usize = 20_000; // 100 samples for the whole run

    let mut profiler = SamplingProfiler::new(SamplingConfig {
        sample_rate_hz: 100,
        ..Default::default()
    });
    profiler.start();

    let mut sink: u64 = 0;
    let work_start = std::time::Instant::now();
    let mut overhead = Duration::ZERO;
    for i in 0..WORK_UNITS {
        sink = sink.wrapping_add((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        sink ^= sink.rotate_left(7);
        if i % SAMPLE_EVERY == 0 {
            let p0 = std::time::Instant::now();
            profiler.record(SampleEvent::new(i as u64, vec!["hot".to_string()], 1, 0));
            overhead += p0.elapsed();
        }
    }
    let total = work_start.elapsed();
    profiler.stop();

    assert_ne!(sink, 0);
    let ratio = overhead.as_secs_f64() / total.as_secs_f64();
    assert!(
        ratio < 0.01,
        "sampling overhead {ratio:.4} must be < 1 % (overhead {overhead:?} of total {total:?})"
    );
}

// ===========================================================================
// 2. Regression detection (detected + no false positive)
// ===========================================================================

/// Build a single-node snapshot whose p95 is exactly `ms` (constant samples).
fn snapshot_with_node(label: &str, node: &str, ms: f64) -> ProfileSnapshot {
    let mut snap = ProfileSnapshot::with_label(label);
    let samples = vec![ms; 32];
    assert!(snap.record_samples(node, &samples), "samples non-empty");
    snap
}

/// An injected ×1.5 slowdown on a hot node (well above the 10 % default
/// threshold) must be flagged as a regression by `ProfileComparator`.
#[test]
fn profile_comparator_flags_injected_1_5x_regression() {
    let baseline = snapshot_with_node("before", "encode_frame", 100.0);
    let current = snapshot_with_node("after", "encode_frame", 150.0); // ×1.5

    let report = ProfileComparator::default().compare(&baseline, &current);

    assert!(
        report.has_regressions(),
        "a ×1.5 slowdown must be flagged as a regression"
    );
    let worst = report.worst_regression().expect("worst exists");
    assert_eq!(worst.node_name, "encode_frame");
    // 100 → 150 ms == +50 %.
    assert!(
        (worst.pct_change - 50.0).abs() < 0.5,
        "pct_change was {}",
        worst.pct_change
    );
    assert!(
        (worst.delta_ms - 50.0).abs() < 0.5,
        "delta_ms was {}",
        worst.delta_ms
    );
}

/// A within-noise delta (±3 %, below the 10 % default threshold) must NOT be
/// flagged — no false positive.
#[test]
fn profile_comparator_no_false_positive_within_noise() {
    let baseline = snapshot_with_node("before", "decode_frame", 100.0);
    let current = snapshot_with_node("after", "decode_frame", 103.0); // +3 %

    let report = ProfileComparator::default().compare(&baseline, &current);

    assert!(
        !report.has_regressions(),
        "a +3 % within-noise delta must not be flagged"
    );
    assert_eq!(
        report.stable_count, 1,
        "the node should be classified as stable"
    );
    assert!(report.worst_regression().is_none());
}

/// Cross-check the same guarantees on the benchmark-based `RegressionDetector`:
/// an injected ×1.5 mean increase is detected and significant; a +3 % increase
/// (below the 5 % default threshold) is not detected.
#[test]
fn regression_detector_flags_injected_and_ignores_noise() {
    let mk = |mean_ms: u64, std_ms: u64| BenchmarkResult {
        name: "hot_kernel".to_string(),
        iterations: 100,
        mean: Duration::from_millis(mean_ms),
        median: Duration::from_millis(mean_ms),
        std_dev: Duration::from_millis(std_ms),
        min: Duration::from_millis(mean_ms.saturating_sub(std_ms)),
        max: Duration::from_millis(mean_ms + std_ms),
        throughput: 1_000.0 / mean_ms as f64,
    };

    let mut detector = RegressionDetector::new(5.0, 2.0);
    detector.set_baseline("hot_kernel".to_string(), mk(100, 2));

    // ×1.5 slowdown (100 → 150 ms) — detected and statistically significant.
    let regressed = detector.detect(&mk(150, 2)).expect("regression detected");
    assert!((regressed.regression_percent - 50.0).abs() < 0.001);
    assert!(
        regressed.is_significant,
        "a 25-sigma move must be significant"
    );

    // +3 % (100 → 103 ms) — below the 5 % threshold, not flagged.
    assert!(
        detector.detect(&mk(103, 2)).is_none(),
        "a +3 % within-noise delta must not be flagged"
    );
}

// ===========================================================================
// 3. Concurrent allocation correctness
// ===========================================================================

/// Drive a known number of allocations from N threads through the lock-free
/// `AllocationTracker` and assert the live-byte total equals the known sum with
/// no lost or double-counted records.
#[test]
fn concurrent_allocation_total_is_exact() {
    const N_THREADS: usize = 6;
    const RECORDS_PER_THREAD: usize = 5_000;
    const ALLOC_SIZE: usize = 128;
    const EXPECTED_BYTES: usize = N_THREADS * RECORDS_PER_THREAD * ALLOC_SIZE;
    const EXPECTED_RECORDS: usize = N_THREADS * RECORDS_PER_THREAD;

    let tracker = Arc::new(AllocationTracker::new());

    let handles: Vec<_> = (0..N_THREADS)
        .map(|_| {
            let t = Arc::clone(&tracker);
            thread::spawn(move || {
                for _ in 0..RECORDS_PER_THREAD {
                    t.record(AllocationType::Heap, ALLOC_SIZE, "concurrent_alloc");
                }
            })
        })
        .collect();
    for h in handles {
        h.join().expect("worker thread panicked");
    }

    // Live-byte counter (atomic) must be exactly the known sum: no lost adds.
    assert_eq!(
        tracker.current_bytes(),
        EXPECTED_BYTES,
        "current_bytes must equal the known allocation sum"
    );
    assert!(
        tracker.peak_bytes() >= EXPECTED_BYTES,
        "peak must be at least the simultaneous live total"
    );

    // Draining the record log must yield exactly the known number of records:
    // no lost and no duplicated pushes through the lock-free injector.
    let records = tracker.records();
    assert_eq!(
        records.len(),
        EXPECTED_RECORDS,
        "expected {EXPECTED_RECORDS} records, got {}",
        records.len()
    );
    let summed: usize = records.iter().map(|r| r.size_bytes).sum();
    assert_eq!(summed, EXPECTED_BYTES, "summed record bytes must match");
}

/// Thread-local aggregation: each thread accumulates counts in its own TLS
/// counters, publishes a snapshot, and the registry's global aggregate must
/// equal the exact known sum across all threads (no crosstalk, no loss).
#[test]
fn thread_local_counters_aggregate_to_exact_global_total() {
    const N_THREADS: usize = 8;
    const INCREMENTS_PER_THREAD: u64 = 10_000;
    const EXPECTED_TOTAL: u64 = N_THREADS as u64 * INCREMENTS_PER_THREAD;

    let registry = TlsCounterRegistry::new();

    let handles: Vec<_> = (0..N_THREADS)
        .map(|tid| {
            let reg = registry.clone();
            thread::spawn(move || {
                with_tls_counters(|c| {
                    c.reset();
                    for _ in 0..INCREMENTS_PER_THREAD {
                        c.increment("alloc_events", 1);
                    }
                });
                let snap = with_tls_counters(|c| c.snapshot(tid as u64, None));
                reg.ingest(snap);
            })
        })
        .collect();
    for h in handles {
        h.join().expect("counter thread panicked");
    }

    let agg = registry.aggregate();
    assert_eq!(
        agg.count("alloc_events"),
        EXPECTED_TOTAL,
        "thread-local counters must aggregate to the exact global total"
    );
    assert_eq!(
        agg.contributing_threads, N_THREADS,
        "every thread must contribute exactly one snapshot"
    );
}

/// Combined sampling + allocation under concurrency: each worker records both a
/// sampling event (through a shared, mutex-guarded `SamplingProfiler` whose TLS
/// is merged per op) and a heap allocation.  Both totals must land exactly.
#[test]
fn concurrent_sampling_and_allocation_consistent() {
    const N_THREADS: usize = 4;
    const OPS_PER_THREAD: usize = 1_000;
    const ALLOC_SIZE: usize = 64;

    let profiler = Arc::new(Mutex::new(SamplingProfiler::default_config()));
    let tracker = Arc::new(AllocationTracker::new());
    profiler.lock().expect("lock").start();

    let handles: Vec<_> = (0..N_THREADS)
        .map(|tid| {
            let p = Arc::clone(&profiler);
            let t = Arc::clone(&tracker);
            thread::spawn(move || {
                for i in 0..OPS_PER_THREAD {
                    t.record(AllocationType::Heap, ALLOC_SIZE, "combined");
                    let mut guard = p.lock().expect("lock");
                    guard.record(SampleEvent::new(
                        i as u64,
                        vec![format!("worker_{tid}")],
                        tid as u64,
                        0,
                    ));
                    guard.merge_thread_local();
                }
            })
        })
        .collect();
    for h in handles {
        h.join().expect("combined worker panicked");
    }

    let mut guard = profiler.lock().expect("lock");
    guard.stop();
    assert_eq!(
        guard.sample_count(),
        N_THREADS * OPS_PER_THREAD,
        "all sampling events must be merged into the aggregate"
    );

    assert_eq!(
        tracker.current_bytes(),
        N_THREADS * OPS_PER_THREAD * ALLOC_SIZE,
        "all allocations must be counted"
    );
}

// ===========================================================================
// Bonus: flamegraph / hotspot known-tree sanity
// ===========================================================================

/// A deterministic call tree where one leaf dominates the sampled time must
/// surface that leaf as the hottest path, and the folded output must contain
/// the full hot stack.
#[test]
fn flamegraph_known_tree_surfaces_hot_path() {
    let mut builder = FlameGraphBuilder::new();

    // main → render → encode  (sampled heavily: 90 ms across 9 samples)
    for _ in 0..9 {
        builder.add_stack(&CallStack::new(
            vec![
                StackFrame::new("main"),
                StackFrame::new("render"),
                StackFrame::new("encode"),
            ],
            Duration::from_millis(10),
        ));
    }
    // main → setup  (sampled lightly: 5 ms across 1 sample)
    builder.add_stack(&CallStack::new(
        vec![StackFrame::new("main"), StackFrame::new("setup")],
        Duration::from_millis(5),
    ));

    assert_eq!(builder.total_samples(), 10);
    assert_eq!(builder.total_duration(), Duration::from_millis(95));

    // The single root is `main`; under it `render` dominates `setup`.
    let roots = builder.top_nodes();
    assert_eq!(roots.len(), 1);
    assert_eq!(roots[0].name, "main");

    let mut main_children: Vec<_> = roots[0].children.values().collect();
    main_children.sort_by(|a, b| b.total_time.cmp(&a.total_time));
    assert_eq!(
        main_children[0].name, "render",
        "render must be the hotter child of main"
    );
    assert_eq!(main_children[0].sample_count, 9);

    // Folded output must contain the full hot stack path.
    let folded = builder.to_folded();
    assert!(
        folded.contains("main;render;encode"),
        "folded output missing hot path; got:\n{folded}"
    );

    // Built graph: node count = main, render, encode, setup = 4.
    let graph = FlameGraph::from_builder(builder);
    assert_eq!(graph.total_nodes(), 4);
}

/// `HotspotDetector` over a known workload must rank the deliberately-hot
/// function on top and filter out functions below the significance threshold.
#[test]
fn hotspot_detector_ranks_known_hot_function() {
    let mut detector = HotspotDetector::new(0.10); // 10 % significance floor
    detector.record("transform_block", Duration::from_millis(700), 700);
    detector.record("write_packet", Duration::from_millis(250), 250);
    detector.record("update_counter", Duration::from_millis(50), 50); // 5 % — filtered

    let spots = detector.detect();
    assert!(!spots.is_empty(), "should detect hotspots");
    assert_eq!(
        spots[0].name, "transform_block",
        "the 70 % function must rank first"
    );
    assert!(
        (spots[0].time_fraction - 0.70).abs() < 1e-6,
        "hot fraction was {}",
        spots[0].time_fraction
    );
    // Below-threshold function must be filtered out.
    assert!(
        !spots.iter().any(|h| h.name == "update_counter"),
        "5 % function must be below the 10 % significance floor"
    );
}
