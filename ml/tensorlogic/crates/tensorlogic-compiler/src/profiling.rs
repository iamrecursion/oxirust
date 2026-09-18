//! Compilation profiling: track time and resource usage across phases.
//!
//! Instruments the compilation pipeline to provide per-phase timing breakdowns,
//! helping identify bottlenecks and optimize compilation performance.
//!
//! # Overview
//!
//! The profiling module provides three main components:
//!
//! - [`ProfileEntry`]: A single profiling entry for a compilation phase
//! - [`ProfileReport`]: A complete profiling report for a compilation run
//! - [`CompilationProfiler`]: Real-time phase tracker for compilation
//!
//! # Examples
//!
//! ```rust
//! use tensorlogic_compiler::profiling::{CompilationProfiler, ProfileReport, ProfileEntry};
//!
//! let mut profiler = CompilationProfiler::new();
//! profiler.set_input_complexity(500);
//!
//! profiler.begin_phase("parse");
//! profiler.set_items(120);
//! profiler.end_phase();
//!
//! profiler.begin_phase("optimize");
//! profiler.set_items(80);
//! profiler.end_phase();
//!
//! profiler.set_output_size(60);
//! let report = profiler.finish();
//! println!("{}", report.summary());
//! ```

use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// A single profiling entry for a compilation phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileEntry {
    /// Phase name (e.g., "parse", "type_check", "optimize", "codegen")
    pub phase: String,
    /// Duration of this phase in milliseconds
    pub duration_ms: f64,
    /// Number of nodes/expressions processed in this phase
    pub items_processed: usize,
    /// Optional notes about what happened
    pub notes: String,
}

impl ProfileEntry {
    /// Create a new profile entry from a phase name, duration, and item count.
    pub fn new(phase: impl Into<String>, duration: Duration, items: usize) -> Self {
        ProfileEntry {
            phase: phase.into(),
            duration_ms: duration.as_secs_f64() * 1000.0,
            items_processed: items,
            notes: String::new(),
        }
    }

    /// Attach notes to this entry (builder pattern).
    pub fn with_notes(mut self, notes: impl Into<String>) -> Self {
        self.notes = notes.into();
        self
    }

    /// Throughput: items per second.
    ///
    /// Returns `f64::INFINITY` when duration is effectively zero.
    pub fn throughput(&self) -> f64 {
        if self.duration_ms < 1e-9 {
            return f64::INFINITY;
        }
        self.items_processed as f64 / (self.duration_ms / 1000.0)
    }
}

/// A complete profiling report for a compilation run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProfileReport {
    /// Individual phase entries in execution order
    pub entries: Vec<ProfileEntry>,
    /// Total wall-clock duration in milliseconds
    pub total_duration_ms: f64,
    /// Expression complexity (number of nodes in input)
    pub input_complexity: usize,
    /// Output graph size (number of nodes in compiled graph)
    pub output_size: usize,
}

impl ProfileReport {
    /// Create a new empty report.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an entry and accumulate its duration into the total.
    pub fn add_entry(&mut self, entry: ProfileEntry) {
        self.total_duration_ms += entry.duration_ms;
        self.entries.push(entry);
    }

    /// Get the slowest phase by duration.
    pub fn slowest_phase(&self) -> Option<&ProfileEntry> {
        self.entries.iter().max_by(|a, b| {
            a.duration_ms
                .partial_cmp(&b.duration_ms)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Get the fastest phase by duration.
    pub fn fastest_phase(&self) -> Option<&ProfileEntry> {
        self.entries.iter().min_by(|a, b| {
            a.duration_ms
                .partial_cmp(&b.duration_ms)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Percentage of total time spent in each phase.
    ///
    /// Returns an empty vector when total duration is effectively zero.
    pub fn phase_percentages(&self) -> Vec<(&str, f64)> {
        if self.total_duration_ms < 1e-9 {
            return vec![];
        }
        self.entries
            .iter()
            .map(|e| {
                (
                    e.phase.as_str(),
                    e.duration_ms / self.total_duration_ms * 100.0,
                )
            })
            .collect()
    }

    /// Number of phases recorded.
    pub fn phase_count(&self) -> usize {
        self.entries.len()
    }

    /// Human-readable summary of the compilation profile.
    pub fn summary(&self) -> String {
        let mut s = format!(
            "Compilation Profile ({:.2}ms total, {} phases):\n",
            self.total_duration_ms,
            self.entries.len()
        );
        for entry in &self.entries {
            let pct = if self.total_duration_ms > 0.0 {
                entry.duration_ms / self.total_duration_ms * 100.0
            } else {
                0.0
            };
            s.push_str(&format!(
                "  {:20} {:8.2}ms ({:5.1}%) [{} items]\n",
                entry.phase, entry.duration_ms, pct, entry.items_processed
            ));
        }
        s.push_str(&format!(
            "Input complexity: {}, Output size: {}\n",
            self.input_complexity, self.output_size
        ));
        s
    }

    /// Compilation speed: output nodes per millisecond.
    pub fn compilation_speed(&self) -> f64 {
        if self.total_duration_ms < 1e-9 {
            return 0.0;
        }
        self.output_size as f64 / self.total_duration_ms
    }
}

/// A profiler that tracks compilation phases in real-time.
///
/// Use [`begin_phase`](CompilationProfiler::begin_phase) /
/// [`end_phase`](CompilationProfiler::end_phase) to bracket each compilation
/// stage, then call [`finish`](CompilationProfiler::finish) to collect the
/// completed [`ProfileReport`].
pub struct CompilationProfiler {
    report: ProfileReport,
    current_phase: Option<(String, Instant, usize)>,
}

impl CompilationProfiler {
    /// Create a new profiler.
    pub fn new() -> Self {
        CompilationProfiler {
            report: ProfileReport::new(),
            current_phase: None,
        }
    }

    /// Start timing a new phase.
    ///
    /// If there is an active phase it is ended automatically before the new
    /// one begins.
    pub fn begin_phase(&mut self, phase: impl Into<String>) {
        // If there's an active phase, end it first
        self.end_phase();
        self.current_phase = Some((phase.into(), Instant::now(), 0));
    }

    /// Set the items processed count for the current phase.
    pub fn set_items(&mut self, count: usize) {
        if let Some((_, _, ref mut items)) = self.current_phase {
            *items = count;
        }
    }

    /// End the current phase and record it.
    pub fn end_phase(&mut self) {
        if let Some((name, start, items)) = self.current_phase.take() {
            let duration = start.elapsed();
            self.report
                .add_entry(ProfileEntry::new(name, duration, items));
        }
    }

    /// Set input complexity on the report.
    pub fn set_input_complexity(&mut self, complexity: usize) {
        self.report.input_complexity = complexity;
    }

    /// Set output graph size on the report.
    pub fn set_output_size(&mut self, size: usize) {
        self.report.output_size = size;
    }

    /// Finish profiling and return the report.
    ///
    /// Any open phase is ended automatically.
    pub fn finish(mut self) -> ProfileReport {
        self.end_phase(); // end any open phase
        self.report
    }

    /// Get a reference to the current (in-progress) report.
    pub fn current_report(&self) -> &ProfileReport {
        &self.report
    }
}

impl Default for CompilationProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Profile a closure as a single phase and return (result, entry).
///
/// The returned [`ProfileEntry`] has `items_processed` set to 0; callers can
/// adjust it or attach notes afterwards.
///
/// # Examples
///
/// ```rust
/// use tensorlogic_compiler::profiling::profile_phase;
///
/// let (result, entry) = profile_phase("my_phase", || {
///     // expensive work
///     42
/// });
/// assert_eq!(result, 42);
/// assert_eq!(entry.phase, "my_phase");
/// ```
pub fn profile_phase<T, F: FnOnce() -> T>(phase: &str, f: F) -> (T, ProfileEntry) {
    let start = Instant::now();
    let result = f();
    let duration = start.elapsed();
    (result, ProfileEntry::new(phase, duration, 0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_profile_entry_new() {
        let entry = ProfileEntry::new("parse", Duration::from_millis(15), 200);
        assert_eq!(entry.phase, "parse");
        assert!((entry.duration_ms - 15.0).abs() < 0.01);
        assert_eq!(entry.items_processed, 200);
        assert!(entry.notes.is_empty());
    }

    #[test]
    fn test_profile_entry_throughput() {
        // 100 items in 10ms = 10_000 items/s
        let entry = ProfileEntry::new("phase", Duration::from_millis(10), 100);
        let tp = entry.throughput();
        assert!((tp - 10_000.0).abs() < 1.0, "expected ~10000, got {}", tp);
    }

    #[test]
    fn test_profile_entry_with_notes() {
        let entry =
            ProfileEntry::new("opt", Duration::from_millis(5), 10).with_notes("applied CSE");
        assert_eq!(entry.notes, "applied CSE");
    }

    #[test]
    fn test_profile_entry_zero_duration_throughput() {
        let entry = ProfileEntry::new("instant", Duration::from_secs(0), 50);
        assert!(entry.throughput().is_infinite());
    }

    #[test]
    fn test_profile_report_new() {
        let r = ProfileReport::new();
        assert!(r.entries.is_empty());
        assert!((r.total_duration_ms).abs() < 1e-9);
        assert_eq!(r.input_complexity, 0);
        assert_eq!(r.output_size, 0);
    }

    #[test]
    fn test_profile_report_add_entry() {
        let mut r = ProfileReport::new();
        r.add_entry(ProfileEntry::new("a", Duration::from_millis(10), 1));
        r.add_entry(ProfileEntry::new("b", Duration::from_millis(20), 2));
        assert_eq!(r.entries.len(), 2);
        assert!((r.total_duration_ms - 30.0).abs() < 0.01);
    }

    #[test]
    fn test_profile_report_slowest() {
        let mut r = ProfileReport::new();
        r.add_entry(ProfileEntry::new("fast", Duration::from_millis(5), 0));
        r.add_entry(ProfileEntry::new("slow", Duration::from_millis(50), 0));
        r.add_entry(ProfileEntry::new("mid", Duration::from_millis(20), 0));
        let slowest = r.slowest_phase().expect("should have slowest");
        assert_eq!(slowest.phase, "slow");
    }

    #[test]
    fn test_profile_report_fastest() {
        let mut r = ProfileReport::new();
        r.add_entry(ProfileEntry::new("fast", Duration::from_millis(5), 0));
        r.add_entry(ProfileEntry::new("slow", Duration::from_millis(50), 0));
        let fastest = r.fastest_phase().expect("should have fastest");
        assert_eq!(fastest.phase, "fast");
    }

    #[test]
    fn test_profile_report_percentages() {
        let mut r = ProfileReport::new();
        r.add_entry(ProfileEntry::new("a", Duration::from_millis(25), 0));
        r.add_entry(ProfileEntry::new("b", Duration::from_millis(75), 0));
        let pcts = r.phase_percentages();
        assert_eq!(pcts.len(), 2);
        let sum: f64 = pcts.iter().map(|(_, p)| p).sum();
        assert!(
            (sum - 100.0).abs() < 0.01,
            "percentages should sum to ~100, got {}",
            sum
        );
        // "a" should be ~25%, "b" should be ~75%
        assert!((pcts[0].1 - 25.0).abs() < 0.1);
        assert!((pcts[1].1 - 75.0).abs() < 0.1);
    }

    #[test]
    fn test_profile_report_phase_count() {
        let mut r = ProfileReport::new();
        assert_eq!(r.phase_count(), 0);
        r.add_entry(ProfileEntry::new("x", Duration::from_millis(1), 0));
        r.add_entry(ProfileEntry::new("y", Duration::from_millis(1), 0));
        r.add_entry(ProfileEntry::new("z", Duration::from_millis(1), 0));
        assert_eq!(r.phase_count(), 3);
    }

    #[test]
    fn test_profile_report_summary() {
        let mut r = ProfileReport::new();
        r.input_complexity = 100;
        r.output_size = 50;
        r.add_entry(ProfileEntry::new("parse", Duration::from_millis(10), 80));
        r.add_entry(ProfileEntry::new("codegen", Duration::from_millis(20), 50));
        let summary = r.summary();
        assert!(
            summary.contains("parse"),
            "should contain phase name 'parse'"
        );
        assert!(
            summary.contains("codegen"),
            "should contain phase name 'codegen'"
        );
        assert!(
            summary.contains("Input complexity: 100"),
            "should contain input complexity"
        );
        assert!(
            summary.contains("Output size: 50"),
            "should contain output size"
        );
    }

    #[test]
    fn test_profile_report_compilation_speed() {
        let mut r = ProfileReport::new();
        r.output_size = 200;
        // Manually set total to avoid floating-point drift from add_entry
        r.total_duration_ms = 100.0;
        let speed = r.compilation_speed();
        assert!((speed - 2.0).abs() < 0.01, "expected 2.0, got {}", speed);
    }

    #[test]
    fn test_profiler_begin_end() {
        let mut profiler = CompilationProfiler::new();
        profiler.begin_phase("single");
        thread::sleep(Duration::from_millis(5));
        profiler.set_items(42);
        profiler.end_phase();

        let report = profiler.current_report();
        assert_eq!(report.entries.len(), 1);
        assert_eq!(report.entries[0].phase, "single");
        assert_eq!(report.entries[0].items_processed, 42);
        assert!(report.entries[0].duration_ms > 0.0);
    }

    #[test]
    fn test_profiler_multiple_phases() {
        let mut profiler = CompilationProfiler::new();
        for name in &["parse", "optimize", "codegen"] {
            profiler.begin_phase(*name);
            profiler.set_items(10);
            profiler.end_phase();
        }
        let report = profiler.finish();
        assert_eq!(report.phase_count(), 3);
        assert_eq!(report.entries[0].phase, "parse");
        assert_eq!(report.entries[1].phase, "optimize");
        assert_eq!(report.entries[2].phase, "codegen");
    }

    #[test]
    fn test_profiler_auto_end_on_begin() {
        let mut profiler = CompilationProfiler::new();
        profiler.begin_phase("first");
        profiler.set_items(5);
        // Starting a new phase should auto-end "first"
        profiler.begin_phase("second");
        profiler.end_phase();

        let report = profiler.finish();
        assert_eq!(report.phase_count(), 2);
        assert_eq!(report.entries[0].phase, "first");
        assert_eq!(report.entries[0].items_processed, 5);
        assert_eq!(report.entries[1].phase, "second");
    }

    #[test]
    fn test_profiler_finish() {
        let mut profiler = CompilationProfiler::new();
        profiler.set_input_complexity(300);
        profiler.set_output_size(150);
        profiler.begin_phase("work");
        profiler.set_items(100);
        // finish should auto-end the open phase
        let report = profiler.finish();
        assert_eq!(report.phase_count(), 1);
        assert_eq!(report.input_complexity, 300);
        assert_eq!(report.output_size, 150);
        assert!(report.total_duration_ms >= 0.0);
    }

    #[test]
    fn test_profiler_set_complexity() {
        let mut profiler = CompilationProfiler::new();
        profiler.set_input_complexity(999);
        assert_eq!(profiler.current_report().input_complexity, 999);
    }

    #[test]
    fn test_profile_phase_fn() {
        let (result, entry) = profile_phase("compute", || {
            let mut sum = 0u64;
            for i in 0..1000 {
                sum += i;
            }
            sum
        });
        assert_eq!(result, 499_500);
        assert_eq!(entry.phase, "compute");
        assert_eq!(entry.items_processed, 0);
        assert!(entry.duration_ms >= 0.0);
    }
}
