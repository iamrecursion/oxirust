//! Statistical sampling profiler for low-overhead CPU profiling.
//!
//! Records periodic stack snapshots and aggregates them into a call-frequency
//! histogram without requiring code instrumentation.
//!
//! # Thread-local design
//!
//! Hot-path recording writes to per-thread `thread_local!` buffers — no
//! synchronisation required on the record path.  When `stop()` or
//! `merge_thread_local()` is called, the **current thread's** TLS buffers are
//! drained into the profiler's global aggregate (`samples` Vec and
//! `hit_counts` HashMap).  For a multi-threaded workload each worker thread
//! should call `merge_thread_local()` before the orchestrator thread reads
//! results, or results should be read only from the orchestrator thread
//! after all workers have finished.

use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::HashMap;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Thread-local recording buffers
//
// Each thread accumulates events and hit-counts independently.
// `merge_thread_local()` drains them into the profiler struct.
// ---------------------------------------------------------------------------

thread_local! {
    /// Per-thread sample-event staging buffer.
    static TL_SAMPLES: RefCell<Vec<SampleEvent>> = const { RefCell::new(Vec::new()) };
    /// Per-thread per-function hit-count staging buffer.
    static TL_HIT_COUNTS: RefCell<HashMap<String, u64>> = RefCell::new(HashMap::new());
}

/// A single sample event captured by the sampling profiler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleEvent {
    /// Wall-clock timestamp of the sample (nanoseconds since epoch).
    pub timestamp_ns: u64,
    /// Symbolic call stack at sample time (outermost first).
    pub stack: Vec<String>,
    /// Thread identifier that was sampled.
    pub thread_id: u64,
    /// CPU time consumed by the thread at sample time (µs).
    pub cpu_time_us: u64,
}

impl SampleEvent {
    /// Creates a new sample event.
    #[must_use]
    pub fn new(timestamp_ns: u64, stack: Vec<String>, thread_id: u64, cpu_time_us: u64) -> Self {
        Self {
            timestamp_ns,
            stack,
            thread_id,
            cpu_time_us,
        }
    }

    /// Returns the top-of-stack function name, if any.
    #[must_use]
    pub fn top_frame(&self) -> Option<&str> {
        self.stack.last().map(String::as_str)
    }

    /// Returns the depth of the recorded stack.
    #[must_use]
    pub fn depth(&self) -> usize {
        self.stack.len()
    }
}

/// Configuration for the sampling profiler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingConfig {
    /// Number of samples to collect per second.
    pub sample_rate_hz: u32,
    /// Maximum stack depth to record per sample.
    pub max_stack_depth: usize,
    /// Collect samples across all threads (`true`) or only the calling thread.
    pub all_threads: bool,
    /// Minimum CPU utilisation (0.0–1.0) to start sampling.
    pub min_cpu_threshold: f64,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            sample_rate_hz: 100,
            max_stack_depth: 64,
            all_threads: true,
            min_cpu_threshold: 0.0,
        }
    }
}

impl SamplingConfig {
    /// Creates a high-frequency sampling configuration (1 kHz).
    #[must_use]
    pub fn high_frequency() -> Self {
        Self {
            sample_rate_hz: 1_000,
            max_stack_depth: 32,
            all_threads: true,
            min_cpu_threshold: 0.0,
        }
    }

    /// Creates a low-overhead sampling configuration (10 Hz).
    #[must_use]
    pub fn low_overhead() -> Self {
        Self {
            sample_rate_hz: 10,
            max_stack_depth: 128,
            all_threads: true,
            min_cpu_threshold: 0.05,
        }
    }

    /// Returns the inter-sample interval as a `Duration`.
    #[must_use]
    pub fn sample_interval(&self) -> Duration {
        if self.sample_rate_hz == 0 {
            Duration::from_secs(1)
        } else {
            Duration::from_nanos(1_000_000_000 / u64::from(self.sample_rate_hz))
        }
    }
}

/// Statistical sampling profiler.
///
/// Collects `SampleEvent`s at a configurable rate and provides aggregated
/// call-frequency statistics without source-level instrumentation.
///
/// ## Thread-local buffering
///
/// `record()` writes to the calling thread's TLS buffers with zero
/// synchronisation cost.  Call `merge_thread_local()` (or `stop()`) from a
/// given thread to drain that thread's buffers into the global aggregate
/// stored in `self`.
#[derive(Debug)]
pub struct SamplingProfiler {
    config: SamplingConfig,
    /// Global aggregate — populated by `merge_thread_local()` / `stop()`.
    samples: Vec<SampleEvent>,
    running: bool,
    start_time: Option<Instant>,
    /// Per-function hit counts accumulated from all merged samples.
    hit_counts: HashMap<String, u64>,
}

impl SamplingProfiler {
    /// Creates a new `SamplingProfiler` with the provided configuration.
    #[must_use]
    pub fn new(config: SamplingConfig) -> Self {
        Self {
            config,
            samples: Vec::new(),
            running: false,
            start_time: None,
            hit_counts: HashMap::new(),
        }
    }

    /// Creates a `SamplingProfiler` with default configuration.
    #[must_use]
    pub fn default_config() -> Self {
        Self::new(SamplingConfig::default())
    }

    /// Returns the configured sample rate in Hz.
    #[must_use]
    pub fn sample_rate_hz(&self) -> u32 {
        self.config.sample_rate_hz
    }

    /// Returns the inter-sample interval.
    #[must_use]
    pub fn sample_interval(&self) -> Duration {
        self.config.sample_interval()
    }

    /// Starts the profiler session.
    ///
    /// Clears both the global aggregate and the calling thread's TLS buffers.
    pub fn start(&mut self) {
        self.running = true;
        self.start_time = Some(Instant::now());
        self.samples.clear();
        self.hit_counts.clear();
        // Also clear this thread's TLS staging buffers so a reused profiler
        // starts with a clean slate.
        TL_SAMPLES.with(|s| s.borrow_mut().clear());
        TL_HIT_COUNTS.with(|h| h.borrow_mut().clear());
    }

    /// Stops the profiler session, merging the calling thread's TLS buffers
    /// into the global aggregate first.
    pub fn stop(&mut self) {
        self.merge_thread_local();
        self.running = false;
    }

    /// Drains the **calling thread's** TLS sample/hit-count buffers into the
    /// profiler's global aggregate (`self.samples` / `self.hit_counts`).
    ///
    /// This is a no-op with respect to any other thread's TLS state.  In a
    /// multi-threaded scenario each worker thread should call this method
    /// (or a cooperative flush path) before the orchestrator reads results.
    pub fn merge_thread_local(&mut self) {
        // Drain per-thread samples into global Vec.
        TL_SAMPLES.with(|s| {
            let mut local = s.borrow_mut();
            self.samples.append(&mut *local);
        });
        // Merge per-thread hit-counts into global HashMap.
        TL_HIT_COUNTS.with(|h| {
            let mut local = h.borrow_mut();
            for (key, count) in local.drain() {
                *self.hit_counts.entry(key).or_insert(0) += count;
            }
        });
    }

    /// Returns `true` if the profiler is currently running.
    #[must_use]
    pub fn is_running(&self) -> bool {
        self.running
    }

    /// Records a pre-built `SampleEvent` into the calling thread's TLS buffer.
    ///
    /// Truncates the stack to [`SamplingConfig::max_stack_depth`] and updates
    /// the per-thread hit-count histogram.  No locking is required — each
    /// thread writes to its own private buffer.
    pub fn record(&mut self, mut event: SampleEvent) {
        event.stack.truncate(self.config.max_stack_depth);
        // Write hit-counts into the calling thread's TLS map.
        TL_HIT_COUNTS.with(|h| {
            let mut local = h.borrow_mut();
            for frame in &event.stack {
                *local.entry(frame.clone()).or_insert(0) += 1;
            }
        });
        // Push event into the calling thread's TLS staging buffer.
        TL_SAMPLES.with(|s| s.borrow_mut().push(event));
    }

    /// Returns a reference to all recorded samples in the **global aggregate**.
    ///
    /// Samples written via `record()` on the current thread are moved into
    /// this aggregate by `stop()` / `merge_thread_local()`.
    #[must_use]
    pub fn samples(&self) -> &[SampleEvent] {
        &self.samples
    }

    /// Returns the total number of samples in the global aggregate.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Returns the elapsed profiling duration, if started.
    #[must_use]
    pub fn elapsed(&self) -> Option<Duration> {
        self.start_time.map(|t| t.elapsed())
    }

    /// Returns the hit count for a specific function name from the global
    /// aggregate.
    #[must_use]
    pub fn hit_count(&self, function_name: &str) -> u64 {
        self.hit_counts.get(function_name).copied().unwrap_or(0)
    }

    /// Returns the top N hottest functions sorted by descending hit count.
    #[must_use]
    pub fn top_functions(&self, n: usize) -> Vec<(&str, u64)> {
        let mut entries: Vec<(&str, u64)> = self
            .hit_counts
            .iter()
            .map(|(k, &v)| (k.as_str(), v))
            .collect();
        entries.sort_by(|a, b| b.1.cmp(&a.1));
        entries.truncate(n);
        entries
    }

    /// Returns the estimated sample rate actually achieved (samples/sec).
    ///
    /// Returns `None` if no samples have been recorded or the profiler has not
    /// been started yet.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn achieved_rate(&self) -> Option<f64> {
        let elapsed = self.elapsed()?;
        let secs = elapsed.as_secs_f64();
        if secs <= 0.0 || self.samples.is_empty() {
            return None;
        }
        Some(self.samples.len() as f64 / secs)
    }

    /// Returns the configuration used by this profiler.
    #[must_use]
    pub fn config(&self) -> &SamplingConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_event(stack: &[&str]) -> SampleEvent {
        SampleEvent::new(0, stack.iter().map(|s| s.to_string()).collect(), 1, 0)
    }

    #[test]
    fn test_default_sample_rate() {
        let p = SamplingProfiler::default_config();
        assert_eq!(p.sample_rate_hz(), 100);
    }

    #[test]
    fn test_high_frequency_config() {
        let cfg = SamplingConfig::high_frequency();
        assert_eq!(cfg.sample_rate_hz, 1_000);
    }

    #[test]
    fn test_low_overhead_config() {
        let cfg = SamplingConfig::low_overhead();
        assert_eq!(cfg.sample_rate_hz, 10);
    }

    #[test]
    fn test_sample_interval_100hz() {
        let cfg = SamplingConfig::default();
        assert_eq!(cfg.sample_interval(), Duration::from_millis(10));
    }

    #[test]
    fn test_sample_interval_zero_rate() {
        let cfg = SamplingConfig {
            sample_rate_hz: 0,
            ..Default::default()
        };
        assert_eq!(cfg.sample_interval(), Duration::from_secs(1));
    }

    #[test]
    fn test_start_stop() {
        let mut p = SamplingProfiler::default_config();
        assert!(!p.is_running());
        p.start();
        assert!(p.is_running());
        p.stop();
        assert!(!p.is_running());
    }

    #[test]
    fn test_record_and_count() {
        let mut p = SamplingProfiler::default_config();
        p.start();
        p.record(make_event(&["main", "render", "encode"]));
        p.record(make_event(&["main", "render"]));
        p.stop();
        assert_eq!(p.sample_count(), 2);
        assert_eq!(p.hit_count("main"), 2);
        assert_eq!(p.hit_count("render"), 2);
        assert_eq!(p.hit_count("encode"), 1);
    }

    #[test]
    fn test_hit_count_missing_function() {
        let p = SamplingProfiler::default_config();
        assert_eq!(p.hit_count("nonexistent"), 0);
    }

    #[test]
    fn test_top_functions_ordering() {
        let mut p = SamplingProfiler::default_config();
        p.start();
        for _ in 0..5 {
            p.record(make_event(&["hot"]));
        }
        for _ in 0..2 {
            p.record(make_event(&["warm"]));
        }
        p.record(make_event(&["cold"]));
        p.stop();
        let top = p.top_functions(2);
        assert_eq!(top[0].0, "hot");
        assert_eq!(top[0].1, 5);
        assert_eq!(top[1].0, "warm");
    }

    #[test]
    fn test_stack_depth_truncation() {
        let cfg = SamplingConfig {
            max_stack_depth: 3,
            ..Default::default()
        };
        let mut p = SamplingProfiler::new(cfg);
        p.start();
        p.record(make_event(&["a", "b", "c", "d", "e"]));
        p.stop();
        assert_eq!(p.samples()[0].stack.len(), 3);
    }

    #[test]
    fn test_elapsed_none_before_start() {
        let p = SamplingProfiler::default_config();
        assert!(p.elapsed().is_none());
    }

    #[test]
    fn test_elapsed_some_after_start() {
        let mut p = SamplingProfiler::default_config();
        p.start();
        std::thread::sleep(Duration::from_millis(5));
        assert!(p.elapsed().is_some());
        p.stop();
    }

    #[test]
    fn test_achieved_rate_none_no_samples() {
        let mut p = SamplingProfiler::default_config();
        p.start();
        assert!(p.achieved_rate().is_none());
        p.stop();
    }

    #[test]
    fn test_sample_event_top_frame() {
        let e = make_event(&["main", "render", "encode"]);
        assert_eq!(e.top_frame(), Some("encode"));
    }

    #[test]
    fn test_sample_event_depth() {
        let e = make_event(&["a", "b", "c"]);
        assert_eq!(e.depth(), 3);
    }

    #[test]
    fn test_config_accessor() {
        let cfg = SamplingConfig::high_frequency();
        let p = SamplingProfiler::new(cfg.clone());
        assert_eq!(p.config().sample_rate_hz, cfg.sample_rate_hz);
    }

    // -----------------------------------------------------------------------
    // Thread-local concurrency tests (Wave 15, Slice H)
    // -----------------------------------------------------------------------

    /// Single-threaded round-trip: verify that `record()` writes to TLS and
    /// `merge_thread_local()` drains them into the global aggregate exactly
    /// once.
    #[test]
    fn test_merge_thread_local_drains_to_aggregate() {
        let mut p = SamplingProfiler::default_config();
        p.start();

        // Record 10 events — these go into TLS, not yet visible in aggregate.
        for i in 0..10u64 {
            p.record(SampleEvent::new(
                i,
                vec!["fn_a".to_string(), "fn_b".to_string()],
                1,
                0,
            ));
        }

        // Global aggregate is still empty before explicit merge.
        // (samples() reads the aggregate, not TLS)
        // After stop() — which calls merge_thread_local() — all 10 arrive.
        p.stop();

        assert_eq!(p.sample_count(), 10, "stop() must drain TLS into aggregate");
        assert_eq!(p.hit_count("fn_a"), 10);
        assert_eq!(p.hit_count("fn_b"), 10);
    }

    /// Single-threaded: calling `merge_thread_local()` explicitly then `stop()`
    /// must not double-count.
    #[test]
    fn test_explicit_merge_then_stop_no_double_count() {
        let mut p = SamplingProfiler::default_config();
        p.start();

        for i in 0..5u64 {
            p.record(SampleEvent::new(i, vec!["work".to_string()], 1, 0));
        }

        // First merge — moves 5 events into aggregate.
        p.merge_thread_local();
        assert_eq!(p.sample_count(), 5);
        assert_eq!(p.hit_count("work"), 5);

        // Record 3 more (go into TLS again).
        for i in 5..8u64 {
            p.record(SampleEvent::new(i, vec!["work".to_string()], 1, 0));
        }

        // stop() merges the remaining 3 — total must be 8, not 5+8.
        p.stop();
        assert_eq!(p.sample_count(), 8);
        assert_eq!(p.hit_count("work"), 8);
    }

    /// Multi-threaded: each thread calls `merge_thread_local()` on the shared
    /// `&mut SamplingProfiler` via a `Mutex` wrapper, so the total across all
    /// threads reaches the expected count.
    ///
    /// Because `thread_local!` is per-thread and merging requires `&mut self`,
    /// each thread holds the Mutex for its `record` + `merge` pair.  The sum
    /// of all thread contributions must equal N_THREADS × EVENTS_PER_THREAD.
    #[test]
    fn test_thread_local_sampling_concurrent() {
        use std::sync::{Arc, Mutex};

        const N_THREADS: usize = 4;
        const EVENTS_PER_THREAD: usize = 100;

        let profiler = Arc::new(Mutex::new(SamplingProfiler::default_config()));
        {
            profiler.lock().expect("lock").start();
        }

        let handles: Vec<_> = (0..N_THREADS)
            .map(|tid| {
                let p = Arc::clone(&profiler);
                std::thread::spawn(move || {
                    // Each thread records its events into its own TLS buffer.
                    // We then lock the profiler and merge this thread's TLS
                    // into the global aggregate.
                    for i in 0..EVENTS_PER_THREAD {
                        let event = SampleEvent::new(
                            i as u64,
                            vec![format!("thread_{}", tid)],
                            tid as u64,
                            0,
                        );
                        // Lock → record (writes to this thread's TLS) →
                        // merge (drains this thread's TLS to aggregate) →
                        // unlock so the next iteration can proceed.
                        let mut guard = p.lock().expect("lock");
                        guard.record(event);
                        guard.merge_thread_local();
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().expect("thread panicked");
        }

        let mut guard = profiler.lock().expect("lock");
        // stop() calls merge_thread_local() — TLS is already empty here,
        // so this is a no-op but must not double-count.
        guard.stop();

        let total = guard.sample_count();
        assert_eq!(
            total,
            N_THREADS * EVENTS_PER_THREAD,
            "expected {} total samples, got {}",
            N_THREADS * EVENTS_PER_THREAD,
            total
        );
    }
}

// ===========================================================================
// Adaptive sampling controller
// ===========================================================================

/// Overhead measurement from a single profiling interval.
///
/// Overhead is defined as the fraction of wall time consumed by the profiler
/// itself (sampling, recording, etc.) rather than the target workload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverheadMeasurement {
    /// Observed overhead fraction (0.0–1.0).
    pub fraction: f64,
    /// Actual sample rate achieved during the measurement interval (Hz).
    pub achieved_rate_hz: f64,
    /// Wall-clock duration of the measurement window.
    pub window_duration: Duration,
}

/// Decision produced by the adaptive controller after each adjustment cycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptiveDecision {
    /// Sample rate was reduced because overhead exceeded the threshold.
    Reduced,
    /// Sample rate was increased because overhead is comfortably below the threshold.
    Increased,
    /// Sample rate was left unchanged.
    Unchanged,
    /// Rate is already at the minimum; no further reduction is possible.
    AtMinimum,
    /// Rate is already at the maximum; no further increase is possible.
    AtMaximum,
}

/// Configuration for the adaptive sampling controller.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveConfig {
    /// Target overhead fraction (default 0.01 = 1 %).
    ///
    /// When the measured overhead exceeds this value the controller reduces
    /// the sample rate.  When the overhead is below
    /// `target_overhead * headroom_factor` the rate is increased.
    pub target_overhead: f64,
    /// Headroom factor (default 0.5).
    ///
    /// The controller will attempt to increase the rate only when the
    /// measured overhead is below `target_overhead * headroom_factor`.
    pub headroom_factor: f64,
    /// Minimum sample rate in Hz (default 1 Hz).
    pub min_rate_hz: u32,
    /// Maximum sample rate in Hz (default 10 000 Hz).
    pub max_rate_hz: u32,
    /// Multiplicative step when reducing the rate (default 0.75 — 25 % cut).
    pub reduction_factor: f64,
    /// Multiplicative step when increasing the rate (default 1.25 — 25 % increase).
    pub increase_factor: f64,
    /// Number of consecutive measurement windows before the controller acts.
    ///
    /// Smooths transient spikes; default is 2.
    pub measurement_window_count: usize,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            target_overhead: 0.01,
            headroom_factor: 0.5,
            min_rate_hz: 1,
            max_rate_hz: 10_000,
            reduction_factor: 0.75,
            increase_factor: 1.25,
            measurement_window_count: 2,
        }
    }
}

/// Adaptive sampling rate controller.
///
/// The controller wraps a `SamplingProfiler` (or any sampling pipeline) and
/// adjusts the configured sample rate up or down in response to measured
/// profiling overhead, keeping the overhead below a configurable fraction of
/// total wall time.
///
/// # Algorithm
///
/// 1. The caller notifies the controller of each completed measurement window
///    via [`observe`](Self::observe).
/// 2. After `measurement_window_count` consecutive windows the controller
///    averages the overhead fractions and calls [`adjust`](Self::adjust)
///    to compute a new rate.
/// 3. The caller retrieves the new rate via
///    [`current_rate_hz`](Self::current_rate_hz) and reconfigures its sampler.
///
/// # Example
///
/// ```
/// use oximedia_profiler::sampling_profiler::{AdaptiveConfig, AdaptiveSamplingController,
///     OverheadMeasurement};
/// use std::time::Duration;
///
/// let cfg = AdaptiveConfig {
///     target_overhead: 0.01,
///     min_rate_hz: 10,
///     max_rate_hz: 1_000,
///     ..Default::default()
/// };
/// let mut ctrl = AdaptiveSamplingController::new(100, cfg);
///
/// // Simulate high overhead measurement.
/// let m = OverheadMeasurement {
///     fraction: 0.05,           // 5 % overhead — above the 1 % target
///     achieved_rate_hz: 100.0,
///     window_duration: Duration::from_millis(100),
/// };
/// ctrl.observe(m);
/// ctrl.observe(OverheadMeasurement {
///     fraction: 0.04,
///     achieved_rate_hz: 100.0,
///     window_duration: Duration::from_millis(100),
/// });
/// // After two windows (measurement_window_count = 2) the controller adjusts.
/// assert!(ctrl.current_rate_hz() < 100);
/// ```
#[derive(Debug)]
pub struct AdaptiveSamplingController {
    /// Currently effective sample rate in Hz.
    current_rate: u32,
    /// Controller configuration.
    config: AdaptiveConfig,
    /// Pending measurements not yet averaged.
    pending: Vec<OverheadMeasurement>,
    /// History of decisions taken.
    history: Vec<(AdaptiveDecision, u32)>,
    /// Total number of adjustments performed.
    adjustment_count: u64,
}

impl AdaptiveSamplingController {
    /// Creates a new controller with the given initial rate and configuration.
    ///
    /// The initial rate is clamped to `[min_rate_hz, max_rate_hz]`.
    #[must_use]
    pub fn new(initial_rate_hz: u32, config: AdaptiveConfig) -> Self {
        let clamped = initial_rate_hz
            .max(config.min_rate_hz)
            .min(config.max_rate_hz);
        Self {
            current_rate: clamped,
            config,
            pending: Vec::new(),
            history: Vec::new(),
            adjustment_count: 0,
        }
    }

    /// Creates a controller with default configuration and the given initial rate.
    #[must_use]
    pub fn with_defaults(initial_rate_hz: u32) -> Self {
        Self::new(initial_rate_hz, AdaptiveConfig::default())
    }

    /// Records an overhead measurement.
    ///
    /// When `measurement_window_count` measurements have been accumulated,
    /// this method internally calls [`adjust`](Self::adjust) and clears the
    /// pending buffer.  Returns the decision taken if an adjustment occurred,
    /// or `None` if more measurements are needed.
    pub fn observe(&mut self, measurement: OverheadMeasurement) -> Option<AdaptiveDecision> {
        self.pending.push(measurement);
        if self.pending.len() >= self.config.measurement_window_count {
            let decision = self.adjust();
            self.pending.clear();
            Some(decision)
        } else {
            None
        }
    }

    /// Forces an immediate adjustment based on the currently pending
    /// measurements, even if `measurement_window_count` has not been reached.
    ///
    /// If no measurements are pending, returns `AdaptiveDecision::Unchanged`.
    pub fn adjust(&mut self) -> AdaptiveDecision {
        if self.pending.is_empty() {
            return AdaptiveDecision::Unchanged;
        }

        // Average overhead fraction across pending windows.
        let avg_overhead: f64 =
            self.pending.iter().map(|m| m.fraction).sum::<f64>() / self.pending.len() as f64;

        let decision = if avg_overhead > self.config.target_overhead {
            self.reduce_rate()
        } else if avg_overhead < self.config.target_overhead * self.config.headroom_factor {
            self.increase_rate()
        } else {
            AdaptiveDecision::Unchanged
        };

        self.history.push((decision, self.current_rate));
        if decision != AdaptiveDecision::Unchanged {
            self.adjustment_count += 1;
        }
        decision
    }

    /// Returns the current effective sample rate in Hz.
    #[must_use]
    pub fn current_rate_hz(&self) -> u32 {
        self.current_rate
    }

    /// Returns the number of adjustments made so far.
    #[must_use]
    pub fn adjustment_count(&self) -> u64 {
        self.adjustment_count
    }

    /// Returns the full history of decisions and resulting rates.
    #[must_use]
    pub fn history(&self) -> &[(AdaptiveDecision, u32)] {
        &self.history
    }

    /// Returns the controller configuration.
    #[must_use]
    pub fn config(&self) -> &AdaptiveConfig {
        &self.config
    }

    /// Resets the controller state: clears pending measurements and history,
    /// but retains the current rate and configuration.
    pub fn reset(&mut self) {
        self.pending.clear();
        self.history.clear();
        self.adjustment_count = 0;
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    fn reduce_rate(&mut self) -> AdaptiveDecision {
        if self.current_rate <= self.config.min_rate_hz {
            return AdaptiveDecision::AtMinimum;
        }
        let new_rate = ((self.current_rate as f64 * self.config.reduction_factor) as u32)
            .max(self.config.min_rate_hz);
        self.current_rate = new_rate;
        AdaptiveDecision::Reduced
    }

    fn increase_rate(&mut self) -> AdaptiveDecision {
        if self.current_rate >= self.config.max_rate_hz {
            return AdaptiveDecision::AtMaximum;
        }
        let new_rate = ((self.current_rate as f64 * self.config.increase_factor) as u32)
            .min(self.config.max_rate_hz);
        self.current_rate = new_rate;
        AdaptiveDecision::Increased
    }
}

// ---------------------------------------------------------------------------
// Adaptive sampling tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod adaptive_tests {
    use super::*;

    fn high_overhead(fraction: f64) -> OverheadMeasurement {
        OverheadMeasurement {
            fraction,
            achieved_rate_hz: 100.0,
            window_duration: Duration::from_millis(100),
        }
    }

    fn low_overhead(fraction: f64) -> OverheadMeasurement {
        OverheadMeasurement {
            fraction,
            achieved_rate_hz: 100.0,
            window_duration: Duration::from_millis(100),
        }
    }

    #[test]
    fn test_initial_rate_clamped_to_bounds() {
        let cfg = AdaptiveConfig {
            min_rate_hz: 10,
            max_rate_hz: 500,
            ..Default::default()
        };
        let ctrl = AdaptiveSamplingController::new(5, cfg);
        assert_eq!(ctrl.current_rate_hz(), 10);
    }

    #[test]
    fn test_rate_reduced_on_high_overhead() {
        let cfg = AdaptiveConfig {
            target_overhead: 0.01,
            measurement_window_count: 1,
            reduction_factor: 0.5,
            min_rate_hz: 1,
            max_rate_hz: 10_000,
            ..Default::default()
        };
        let mut ctrl = AdaptiveSamplingController::new(100, cfg);
        let decision = ctrl.observe(high_overhead(0.05)).expect("should decide");
        assert_eq!(decision, AdaptiveDecision::Reduced);
        assert!(ctrl.current_rate_hz() < 100, "rate should have fallen");
    }

    #[test]
    fn test_rate_increased_on_low_overhead() {
        let cfg = AdaptiveConfig {
            target_overhead: 0.10,
            headroom_factor: 0.5, // increase when overhead < 5%
            measurement_window_count: 1,
            increase_factor: 2.0,
            min_rate_hz: 1,
            max_rate_hz: 10_000,
            ..Default::default()
        };
        let mut ctrl = AdaptiveSamplingController::new(50, cfg);
        let decision = ctrl.observe(low_overhead(0.001)).expect("should decide");
        assert_eq!(decision, AdaptiveDecision::Increased);
        assert!(ctrl.current_rate_hz() > 50, "rate should have risen");
    }

    #[test]
    fn test_unchanged_when_overhead_in_range() {
        let cfg = AdaptiveConfig {
            target_overhead: 0.05,
            headroom_factor: 0.5,
            measurement_window_count: 1,
            ..Default::default()
        };
        let mut ctrl = AdaptiveSamplingController::new(100, cfg);
        // overhead = 3 % — below 5 % but above 2.5 % (headroom), so unchanged.
        let decision = ctrl.observe(high_overhead(0.03)).expect("should decide");
        assert_eq!(decision, AdaptiveDecision::Unchanged);
        assert_eq!(ctrl.current_rate_hz(), 100);
    }

    #[test]
    fn test_at_minimum_guard() {
        let cfg = AdaptiveConfig {
            target_overhead: 0.01,
            measurement_window_count: 1,
            min_rate_hz: 100,
            max_rate_hz: 10_000,
            ..Default::default()
        };
        let mut ctrl = AdaptiveSamplingController::new(100, cfg);
        let decision = ctrl.observe(high_overhead(0.99)).expect("should decide");
        assert_eq!(decision, AdaptiveDecision::AtMinimum);
        assert_eq!(ctrl.current_rate_hz(), 100);
    }

    #[test]
    fn test_at_maximum_guard() {
        let cfg = AdaptiveConfig {
            target_overhead: 0.50,
            headroom_factor: 0.5,
            measurement_window_count: 1,
            min_rate_hz: 1,
            max_rate_hz: 100,
            increase_factor: 2.0,
            ..Default::default()
        };
        let mut ctrl = AdaptiveSamplingController::new(100, cfg);
        let decision = ctrl.observe(low_overhead(0.0)).expect("should decide");
        assert_eq!(decision, AdaptiveDecision::AtMaximum);
        assert_eq!(ctrl.current_rate_hz(), 100);
    }

    #[test]
    fn test_buffering_waits_for_window_count() {
        let cfg = AdaptiveConfig {
            target_overhead: 0.01,
            measurement_window_count: 3,
            reduction_factor: 0.5,
            ..Default::default()
        };
        let mut ctrl = AdaptiveSamplingController::new(100, cfg);
        assert!(ctrl.observe(high_overhead(0.99)).is_none());
        assert!(ctrl.observe(high_overhead(0.99)).is_none());
        // Third observation triggers adjustment.
        let decision = ctrl
            .observe(high_overhead(0.99))
            .expect("should decide now");
        assert_eq!(decision, AdaptiveDecision::Reduced);
    }

    #[test]
    fn test_adjustment_count_increments() {
        let cfg = AdaptiveConfig {
            target_overhead: 0.01,
            measurement_window_count: 1,
            ..Default::default()
        };
        let mut ctrl = AdaptiveSamplingController::new(100, cfg);
        ctrl.observe(high_overhead(0.99));
        ctrl.observe(high_overhead(0.99));
        assert_eq!(ctrl.adjustment_count(), 2);
    }

    #[test]
    fn test_history_records_decisions() {
        let cfg = AdaptiveConfig {
            target_overhead: 0.01,
            measurement_window_count: 1,
            ..Default::default()
        };
        let mut ctrl = AdaptiveSamplingController::new(200, cfg);
        ctrl.observe(high_overhead(0.05));
        ctrl.observe(high_overhead(0.05));
        assert_eq!(ctrl.history().len(), 2);
    }

    #[test]
    fn test_reset_clears_state() {
        let cfg = AdaptiveConfig {
            target_overhead: 0.01,
            measurement_window_count: 1,
            ..Default::default()
        };
        let mut ctrl = AdaptiveSamplingController::new(100, cfg);
        ctrl.observe(high_overhead(0.99));
        let rate_after_reduce = ctrl.current_rate_hz();
        ctrl.reset();
        assert_eq!(ctrl.adjustment_count(), 0);
        assert!(ctrl.history().is_empty());
        // Rate is NOT reset (intentional — the caller decides).
        assert_eq!(ctrl.current_rate_hz(), rate_after_reduce);
    }

    #[test]
    fn test_adjust_without_pending_returns_unchanged() {
        let mut ctrl = AdaptiveSamplingController::with_defaults(100);
        let decision = ctrl.adjust();
        assert_eq!(decision, AdaptiveDecision::Unchanged);
    }

    #[test]
    fn test_multiple_reductions_approach_minimum() {
        let cfg = AdaptiveConfig {
            target_overhead: 0.001,
            measurement_window_count: 1,
            reduction_factor: 0.5,
            min_rate_hz: 5,
            max_rate_hz: 10_000,
            ..Default::default()
        };
        let mut ctrl = AdaptiveSamplingController::new(1_000, cfg);
        for _ in 0..20 {
            ctrl.observe(high_overhead(0.99));
        }
        assert!(ctrl.current_rate_hz() <= 10);
    }

    #[test]
    fn test_with_defaults_constructor() {
        let ctrl = AdaptiveSamplingController::with_defaults(500);
        assert_eq!(ctrl.current_rate_hz(), 500);
        assert_eq!(
            ctrl.config().target_overhead,
            AdaptiveConfig::default().target_overhead
        );
    }
}
