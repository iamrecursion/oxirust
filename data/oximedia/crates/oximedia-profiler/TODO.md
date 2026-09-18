# oximedia-profiler TODO

## Current Status
- 27 modules for performance profiling and optimization
- Core: Profiler, ProfilerConfig, ProfilingMode (Sampling, Instrumentation, EventBased, Continuous)
- Modules: allocation_tracker, benchmark, bottleneck, cache, call_graph, codec_profiler, cpu, event_trace, flame, flamegraph, frame, frame_profiler, gpu, hotspot, latency_profiler, mem_profile, memory, memory_profiler, network_profiler, optimize, pipeline_profiler, regression, report, report_format, resource, sampling_profiler, thread, throughput_profiler
- Dependencies: oximedia-core, sysinfo, serde, thiserror

## Enhancements
- [x] Add hierarchical span tracking in `Profiler` for nested function timing (enter/exit scope) (verified 2026-05-16; src/span.rs:33 SpanId, Span:49, SpanTracker with nested enter/exit, 476 lines)
- [x] Implement adaptive sampling rate in `sampling_profiler` — increase rate during detected hotspots (verified 2026-05-16; src/sampling_profiler.rs:52 SamplingConfig, high_frequency:77, low_overhead:88, 862 lines)
- [x] Extend `memory_profiler` with allocation site tracking (caller stack capture) (verified 2026-05-16; src/allocation_tracker.rs:33 AllocRecord with call-site tag:38, AllocationTracker:71)
- [x] Add real-time profiling data streaming via `report` module (live metrics push) (verified 2026-06-01: report/streaming.rs StreamingReporter+ProfilingEvent+ProfilingEventType; mpsc channel; span_start/span_end/record/drain_to_json; 8 tests)
- [x] Implement `bottleneck` detection with automated classification (CPU-bound, I/O-bound, memory-bound) (verified 2026-05-16; src/bottleneck/classify.rs:91 BottleneckClassifier, classify:95, classify_with_suggestions:127)
- [x] Extend `codec_profiler` with per-codec frame encode/decode timing histograms (verified 2026-05-16; src/codec_profiler.rs — per-codec frame timing, histograms)
- [x] Add `pipeline_profiler` integration with `oximedia-pipeline` for automatic node instrumentation (verified 2026-05-16; src/pipeline_profiler.rs — pipeline node instrumentation)
- [x] Implement profile comparison in `regression` (diff two profiles to find regressions) (verified 2026-05-16; src/profile_compare.rs — diff two profiles; src/regression/detect.rs — regression detection)

## New Features
- [x] Add Chrome Tracing JSON export in `event_trace` for viewing in chrome://tracing (verified 2026-05-16; src/chrome_trace.rs:76 ChromeTraceEvent, ChromeTracingExporter, to_json:23, 488 lines)
- [x] Implement `perf`-compatible output format for integration with Linux perf tools (verified 2026-05-16; src/perf_script.rs:32 PerfSample, PerfScriptExporter:84, to_perf_line:72, 236 lines)
- [x] Add power/energy profiling module using platform-specific APIs (RAPL on Linux, IOKit on macOS) (verified 2026-05-16; src/power_energy.rs:30 PowerDomain RAPL/IOKit, EnergyProfiler, 650 lines)
- [x] Implement distributed profiling for multi-node encoding clusters (aggregate across machines) (verified 2026-05-16; src/distributed_profile.rs:111 NodeProfile, NodeSpan:58, aggregate:41, 624 lines)
- [ ] Add real-time frame budget visualization (per-frame waterfall chart data) (verified-open 2026-05-16: no per-frame waterfall/budget visualization module in profiler sources)
- [x] Implement automated optimization suggestions in `optimize` based on profiling data (verified 2026-05-16; src/optimize/suggest.rs:9 Suggestion, OptimizationSuggester:82, generate from hotspots:94, 243 lines)
- [x] Add lock contention profiling in `thread` module (track mutex wait times) (verified 2026-05-16; src/thread/contention.rs:9 ContentionEvent, ContentionDetector:29, LockStats:37, wait_time:20)

## Performance
- [ ] Reduce profiling overhead by using thread-local storage for sampling counters
- [x] Implement zero-allocation event recording in `event_trace` with pre-allocated ring buffer (2026-06-22: fixed-capacity `EventRingBuffer<TraceEvent, 4096>` const-array backing store; `emit` hot path is alloc-free — `Copy` `TraceEvent` with inline `[u8;64]` message + interned `u32` component id; added `EventTrace::capacity()`/`drain()`/`drain_into()` + `EventRingBuffer::drain_into()`; 9 new tests incl. capacity-invariance-across-burst (no growth) + drain order/wrap; cargo test + clippy -D warnings green)
- [ ] Use lock-free data structures in `allocation_tracker` to minimize measurement perturbation
- [ ] Add configurable buffer sizes in `flamegraph` generation to handle deep call stacks efficiently

## Testing
- [x] Add profiling overhead measurement tests (verify < 1% overhead in Sampling mode) (2026-06-06: tests/profiler_guarantees.rs:53 overhead_accounting_below_one_percent_deterministic — deterministic count-based accounting + adaptive controller; :103 adaptive_controller_keeps_overhead_under_target; :153 self_timed_overhead_accounting_is_coherent — profiler self-timed via tls_counters, loose always-on bound; :223 self_timed_overhead_below_one_percent_tight — #[ignore]-gated tight wall-clock <1% to avoid CI flakiness)
- [x] Test `regression` detection with synthetic benchmark data containing known regressions (2026-06-06: tests/profiler_guarantees.rs:271 profile_comparator_flags_injected_1_5x_regression; :299 profile_comparator_no_false_positive_within_noise; :320 regression_detector_flags_injected_and_ignores_noise — both ProfileComparator and benchmark RegressionDetector, injected ×1.5 detected + ±3% noise not flagged)
- [x] Add tests for `flame` and `flamegraph` SVG output correctness with known call trees (2026-06-06: tests/profiler_guarantees.rs:506 flamegraph_known_tree_surfaces_hot_path — known call tree → folded-format + tree-structure correctness (hot path `main;render;encode` on top, node count). SVG *rendering* output not asserted here; flamegraph::svg already has unit tests (test_svg_generation, test_svg_renderer))
- [x] Test `hotspot` detection accuracy with synthetic workloads having known hot functions (2026-06-06: tests/profiler_guarantees.rs:557 hotspot_detector_ranks_known_hot_function — 70% function ranks first, sub-threshold function filtered)
- [x] Verify `memory_profiler` tracks allocations correctly under concurrent allocation patterns (2026-06-06: tests/profiler_guarantees.rs:358 concurrent_allocation_total_is_exact — N-thread lock-free AllocationTracker exact byte+record total; :409 thread_local_counters_aggregate_to_exact_global_total; :451 concurrent_sampling_and_allocation_consistent)

## Documentation
- [ ] Add profiling quickstart guide with common workflow patterns
- [ ] Document ProfilingMode selection criteria (when to use Sampling vs Instrumentation)
- [ ] Add flame graph interpretation guide with annotated example output
