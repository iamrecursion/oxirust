//! Bottleneck/hotspot detection: hotspot identification, throughput limits, and Amdahl estimates.

#![allow(dead_code)]

use std::collections::HashMap;
use std::time::Duration;

/// Type of bottleneck identified.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BottleneckKind {
    /// CPU-bound: compute is the limiting factor.
    Cpu,
    /// Memory-bound: bandwidth or cache misses dominate.
    Memory,
    /// I/O-bound: disk or network throughput is the bottleneck.
    Io,
    /// Synchronisation-bound: threads waiting on locks or barriers.
    Synchronisation,
    /// GPU-bound: GPU pipeline is saturated.
    Gpu,
    /// Unknown or mixed bottleneck.
    Unknown,
}

impl std::fmt::Display for BottleneckKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu => write!(f, "CPU"),
            Self::Memory => write!(f, "Memory"),
            Self::Io => write!(f, "I/O"),
            Self::Synchronisation => write!(f, "Synchronisation"),
            Self::Gpu => write!(f, "GPU"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

/// A hotspot: a region of code that consumes a significant fraction of time.
#[derive(Debug, Clone)]
pub struct Hotspot {
    /// Name or label of the hotspot (e.g. function name).
    pub name: String,
    /// Kind of bottleneck.
    pub kind: BottleneckKind,
    /// Fraction of total time spent here (0.0–1.0).
    pub time_fraction: f64,
    /// Total time spent here.
    pub total_time: Duration,
    /// Hit count (samples or call count).
    pub hits: u64,
    /// Optional suggestion for optimisation.
    pub suggestion: Option<String>,
}

impl Hotspot {
    /// Create a new hotspot entry.
    pub fn new(
        name: impl Into<String>,
        kind: BottleneckKind,
        total_time: Duration,
        total_profile_time: Duration,
        hits: u64,
    ) -> Self {
        #[allow(clippy::cast_precision_loss)]
        let time_fraction = if total_profile_time.is_zero() {
            0.0
        } else {
            total_time.as_secs_f64() / total_profile_time.as_secs_f64()
        };
        Self {
            name: name.into(),
            kind,
            time_fraction,
            total_time,
            hits,
            suggestion: None,
        }
    }

    /// Attach a human-readable optimisation suggestion.
    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestion = Some(suggestion.into());
        self
    }

    /// Whether this hotspot accounts for more than `threshold` of total time.
    pub fn is_significant(&self, threshold: f64) -> bool {
        self.time_fraction >= threshold
    }
}

/// Amdahl's Law estimate for the theoretical speedup achievable by parallelising
/// a fraction `p` of the workload across `n` processors.
///
/// Formula: S(n) = 1 / ((1 - p) + p / n)
#[allow(clippy::cast_precision_loss)]
pub fn amdahl_speedup(parallel_fraction: f64, num_processors: usize) -> f64 {
    if num_processors == 0 {
        return 1.0;
    }
    let n = num_processors as f64;
    let p = parallel_fraction.clamp(0.0, 1.0);
    1.0 / ((1.0 - p) + p / n)
}

/// Maximum theoretical speedup as processor count approaches infinity.
pub fn amdahl_max_speedup(parallel_fraction: f64) -> f64 {
    let p = parallel_fraction.clamp(0.0, 1.0);
    if (1.0 - p).abs() < f64::EPSILON {
        f64::INFINITY
    } else {
        1.0 / (1.0 - p)
    }
}

/// Throughput analyser: records per-stage throughput and identifies the
/// stage that limits overall throughput (the pipeline bottleneck).
#[derive(Debug, Default)]
pub struct ThroughputAnalyser {
    /// Stage name -> (items processed, elapsed time)
    stages: HashMap<String, (u64, Duration)>,
}

impl ThroughputAnalyser {
    /// Create a new analyser.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record throughput for a pipeline stage.
    pub fn record_stage(&mut self, name: impl Into<String>, items: u64, elapsed: Duration) {
        self.stages.insert(name.into(), (items, elapsed));
    }

    /// Throughput of a stage in items/second. Returns `None` if stage unknown.
    #[allow(clippy::cast_precision_loss)]
    pub fn stage_throughput(&self, name: &str) -> Option<f64> {
        self.stages.get(name).map(|(items, elapsed)| {
            if elapsed.is_zero() {
                f64::INFINITY
            } else {
                *items as f64 / elapsed.as_secs_f64()
            }
        })
    }

    /// Name and throughput of the slowest (bottleneck) stage.
    pub fn bottleneck_stage(&self) -> Option<(String, f64)> {
        self.stages
            .iter()
            .filter_map(|(name, (items, elapsed))| {
                if elapsed.is_zero() {
                    return None;
                }
                #[allow(clippy::cast_precision_loss)]
                let tp = *items as f64 / elapsed.as_secs_f64();
                Some((name.clone(), tp))
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Overall pipeline throughput limited by the bottleneck stage.
    pub fn pipeline_throughput(&self) -> f64 {
        self.bottleneck_stage().map(|(_, tp)| tp).unwrap_or(0.0)
    }
}

/// Hotspot detector: identifies the most time-consuming functions from a
/// map of function names to `(total_time, hit_count)`.
#[derive(Debug)]
pub struct HotspotDetector {
    /// Significance threshold (0.0–1.0). Hotspots below this fraction are ignored.
    pub threshold: f64,
    entries: HashMap<String, (Duration, u64)>,
}

impl HotspotDetector {
    /// Create a new detector with the given significance threshold.
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold: threshold.clamp(0.0, 1.0),
            entries: HashMap::new(),
        }
    }

    /// Record profiling data for a function.
    pub fn record(&mut self, name: impl Into<String>, time: Duration, hits: u64) {
        let entry = self.entries.entry(name.into()).or_default();
        entry.0 += time;
        entry.1 += hits;
    }

    /// Total time across all recorded entries.
    pub fn total_time(&self) -> Duration {
        self.entries.values().map(|(t, _)| *t).sum()
    }

    /// Detect hotspots above the threshold, sorted by time fraction descending.
    pub fn detect(&self) -> Vec<Hotspot> {
        let total = self.total_time();
        let mut hotspots: Vec<Hotspot> = self
            .entries
            .iter()
            .map(|(name, (time, hits))| {
                Hotspot::new(name.clone(), BottleneckKind::Unknown, *time, total, *hits)
            })
            .filter(|h| h.is_significant(self.threshold))
            .collect();
        hotspots.sort_by(|a, b| {
            b.time_fraction
                .partial_cmp(&a.time_fraction)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hotspots
    }

    /// Top N hotspots by time fraction.
    pub fn top(&self, n: usize) -> Vec<Hotspot> {
        self.detect().into_iter().take(n).collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bottleneck_kind_display() {
        assert_eq!(BottleneckKind::Cpu.to_string(), "CPU");
        assert_eq!(BottleneckKind::Io.to_string(), "I/O");
        assert_eq!(
            BottleneckKind::Synchronisation.to_string(),
            "Synchronisation"
        );
    }

    #[test]
    fn test_hotspot_time_fraction() {
        let hs = Hotspot::new(
            "render",
            BottleneckKind::Cpu,
            Duration::from_millis(500),
            Duration::from_secs(1),
            10,
        );
        assert!((hs.time_fraction - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_hotspot_time_fraction_zero_total() {
        let hs = Hotspot::new(
            "x",
            BottleneckKind::Unknown,
            Duration::ZERO,
            Duration::ZERO,
            0,
        );
        assert_eq!(hs.time_fraction, 0.0);
    }

    #[test]
    fn test_hotspot_is_significant() {
        let hs = Hotspot::new(
            "foo",
            BottleneckKind::Memory,
            Duration::from_millis(600),
            Duration::from_secs(1),
            5,
        );
        assert!(hs.is_significant(0.5));
        assert!(!hs.is_significant(0.7));
    }

    #[test]
    fn test_hotspot_with_suggestion() {
        let hs = Hotspot::new(
            "bar",
            BottleneckKind::Cpu,
            Duration::ZERO,
            Duration::ZERO,
            0,
        )
        .with_suggestion("Use SIMD");
        assert_eq!(hs.suggestion.as_deref(), Some("Use SIMD"));
    }

    #[test]
    fn test_amdahl_speedup_100pct_parallel() {
        // 100% parallel, 4 cores → speedup = 4
        let s = amdahl_speedup(1.0, 4);
        assert!((s - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_amdahl_speedup_0pct_parallel() {
        // 0% parallel → speedup always 1
        let s = amdahl_speedup(0.0, 16);
        assert!((s - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_amdahl_speedup_zero_processors() {
        assert_eq!(amdahl_speedup(0.9, 0), 1.0);
    }

    #[test]
    fn test_amdahl_max_speedup() {
        // 90% parallel → max speedup = 10
        let s = amdahl_max_speedup(0.9);
        assert!((s - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_amdahl_max_speedup_fully_parallel() {
        assert!(amdahl_max_speedup(1.0).is_infinite());
    }

    #[test]
    fn test_throughput_analyser_stage_throughput() {
        let mut ta = ThroughputAnalyser::new();
        ta.record_stage("decode", 100, Duration::from_secs(1));
        let tp = ta
            .stage_throughput("decode")
            .expect("should succeed in test");
        assert!((tp - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_throughput_analyser_unknown_stage() {
        let ta = ThroughputAnalyser::new();
        assert!(ta.stage_throughput("missing").is_none());
    }

    #[test]
    fn test_throughput_analyser_bottleneck() {
        let mut ta = ThroughputAnalyser::new();
        ta.record_stage("fast", 1000, Duration::from_secs(1));
        ta.record_stage("slow", 10, Duration::from_secs(1));
        let (name, _tp) = ta.bottleneck_stage().expect("should succeed in test");
        assert_eq!(name, "slow");
    }

    #[test]
    fn test_hotspot_detector_detects_above_threshold() {
        let mut hd = HotspotDetector::new(0.3);
        hd.record("heavy", Duration::from_millis(800), 10);
        hd.record("light", Duration::from_millis(200), 5);
        let spots = hd.detect();
        assert!(spots.iter().any(|h| h.name == "heavy"));
        // "light" is 20% — below 30% threshold
        assert!(!spots.iter().any(|h| h.name == "light"));
    }

    #[test]
    fn test_hotspot_detector_top_n() {
        let mut hd = HotspotDetector::new(0.0);
        hd.record("a", Duration::from_millis(300), 1);
        hd.record("b", Duration::from_millis(500), 1);
        hd.record("c", Duration::from_millis(200), 1);
        let top = hd.top(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].name, "b");
    }

    #[test]
    fn test_hotspot_detector_total_time() {
        let mut hd = HotspotDetector::new(0.0);
        hd.record("x", Duration::from_millis(400), 1);
        hd.record("y", Duration::from_millis(600), 1);
        assert_eq!(hd.total_time(), Duration::from_secs(1));
    }
}
