//! Bottleneck classification for hierarchical span trees.
//!
//! This module identifies and labels the slowest spans from a `SpanTracker`,
//! classifying them into categories such as CPU-bound, I/O-bound, or
//! memory-bound based on configurable heuristics applied to span duration,
//! depth, and descendant overhead.
//!
//! # Design
//!
//! The classifier works in three passes:
//!
//! 1. **Aggregation** — collect all closed spans from the `SpanTracker`,
//!    computing self-time (exclusive of children) for each span.
//! 2. **Ranking** — sort by self-time descending to find the "hottest" nodes.
//! 3. **Labelling** — apply heuristics based on name patterns, duration
//!    thresholds, and depth position to assign a `SpanBottleneckKind`.
//!
//! # Example
//!
//! ```
//! use oximedia_profiler::span::SpanTracker;
//! use oximedia_profiler::span_bottleneck::SpanBottleneckClassifier;
//!
//! let tracker = SpanTracker::new();
//! {
//!     let _root = tracker.enter("decode");
//!     {
//!         let _inner = tracker.enter("parse_headers");
//!     }
//! }
//!
//! let classifier = SpanBottleneckClassifier::new();
//! let report = classifier.classify(&tracker, 5);
//! println!("{}", report.summary());
//! ```

#![allow(dead_code)]

use crate::span::{Span, SpanId, SpanTracker};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

// ---------------------------------------------------------------------------
// SpanBottleneckKind
// ---------------------------------------------------------------------------

/// Classification assigned to a slow span.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SpanBottleneckKind {
    /// The span dominates CPU time (long self-time, shallow / leaf position).
    CpuHotspot,
    /// The span is likely waiting on I/O (name hints or long duration with
    /// low child count).
    IoWait,
    /// The span is a deep ancestor whose total time dwarfs its self-time,
    /// suggesting it aggregates many short descendant calls.
    Aggregator,
    /// The span has significant self-time that cannot be attributed to a more
    /// specific category.
    SelfTimeHeavy,
    /// A leaf span with non-trivial self-time; potential micro-hotspot.
    LeafHotspot,
}

impl SpanBottleneckKind {
    /// Human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::CpuHotspot => "CPU hotspot",
            Self::IoWait => "I/O wait",
            Self::Aggregator => "Aggregator (high child overhead)",
            Self::SelfTimeHeavy => "Self-time heavy",
            Self::LeafHotspot => "Leaf hotspot",
        }
    }
}

// ---------------------------------------------------------------------------
// SpanBottleneck
// ---------------------------------------------------------------------------

/// A single classified bottleneck span.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanBottleneck {
    /// The span identifier.
    pub span_id: SpanId,
    /// Human-readable span name.
    pub name: String,
    /// Total wall-clock duration of the span (inclusive of children).
    pub total_duration: Duration,
    /// Duration attributed to this span alone (total minus sum of children).
    pub self_duration: Duration,
    /// Number of direct children.
    pub child_count: usize,
    /// Nesting depth from a root (0 = root).
    pub depth: usize,
    /// Assigned bottleneck kind.
    pub kind: SpanBottleneckKind,
    /// Fraction of overall profiling time consumed (0.0–1.0).
    pub fraction_of_total: f64,
}

impl SpanBottleneck {
    /// Returns a one-line description suitable for logging.
    #[must_use]
    pub fn describe(&self) -> String {
        format!(
            "[{}] \"{}\" — self: {:?}, total: {:?}, {:.1}% of wall time (depth {})",
            self.kind.label(),
            self.name,
            self.self_duration,
            self.total_duration,
            self.fraction_of_total * 100.0,
            self.depth,
        )
    }
}

// ---------------------------------------------------------------------------
// SpanBottleneckReport
// ---------------------------------------------------------------------------

/// Full classification report produced by `SpanBottleneckClassifier`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanBottleneckReport {
    /// Top-N bottlenecks ordered by self-time descending.
    pub bottlenecks: Vec<SpanBottleneck>,
    /// The total wall-clock duration used as the denominator for fractions.
    pub wall_time: Duration,
    /// Total span count analysed.
    pub span_count: usize,
}

impl SpanBottleneckReport {
    /// Returns a multi-line human-readable summary.
    #[must_use]
    pub fn summary(&self) -> String {
        let mut out = format!(
            "=== Span Bottleneck Report ({} spans, wall time: {:?}) ===\n",
            self.span_count, self.wall_time
        );
        if self.bottlenecks.is_empty() {
            out.push_str("  (no bottlenecks found)\n");
        } else {
            for (i, b) in self.bottlenecks.iter().enumerate() {
                out.push_str(&format!("  {:2}. {}\n", i + 1, b.describe()));
            }
        }
        out
    }

    /// Returns the primary (worst) bottleneck, if any.
    #[must_use]
    pub fn primary(&self) -> Option<&SpanBottleneck> {
        self.bottlenecks.first()
    }
}

// ---------------------------------------------------------------------------
// Classifier configuration
// ---------------------------------------------------------------------------

/// Thresholds for the span bottleneck classifier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanBottleneckConfig {
    /// Self-time fraction of wall time above which a span is a hotspot
    /// (default: 0.05 = 5 %).
    pub hotspot_fraction: f64,
    /// Fraction of total time spent in children above which a span is
    /// classified as an aggregator (default: 0.85 = 85 %).
    pub aggregator_child_fraction: f64,
    /// Minimum absolute self-time to qualify as a bottleneck (default: 1 µs).
    pub min_self_time: Duration,
    /// Name substrings that hint at I/O activity.
    pub io_name_hints: Vec<String>,
}

impl Default for SpanBottleneckConfig {
    fn default() -> Self {
        Self {
            hotspot_fraction: 0.05,
            aggregator_child_fraction: 0.85,
            min_self_time: Duration::from_micros(1),
            io_name_hints: vec![
                "read".to_owned(),
                "write".to_owned(),
                "io".to_owned(),
                "fetch".to_owned(),
                "net".to_owned(),
                "recv".to_owned(),
                "send".to_owned(),
                "socket".to_owned(),
                "file".to_owned(),
                "disk".to_owned(),
                "http".to_owned(),
                "tcp".to_owned(),
            ],
        }
    }
}

// ---------------------------------------------------------------------------
// Classifier
// ---------------------------------------------------------------------------

/// Classifies the slowest spans in a `SpanTracker` into bottleneck categories.
#[derive(Debug, Clone)]
pub struct SpanBottleneckClassifier {
    config: SpanBottleneckConfig,
}

impl SpanBottleneckClassifier {
    /// Creates a classifier with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: SpanBottleneckConfig::default(),
        }
    }

    /// Creates a classifier with custom configuration.
    #[must_use]
    pub fn with_config(config: SpanBottleneckConfig) -> Self {
        Self { config }
    }

    /// Analyses all closed spans in `tracker` and returns the top `top_n`
    /// bottleneck spans.
    ///
    /// Open (unclosed) spans are silently excluded.
    #[must_use]
    pub fn classify(&self, tracker: &SpanTracker, top_n: usize) -> SpanBottleneckReport {
        let all_spans = tracker.all_spans();
        let closed: Vec<&Span> = all_spans.iter().filter(|s| s.is_closed()).collect();

        if closed.is_empty() {
            return SpanBottleneckReport {
                bottlenecks: Vec::new(),
                wall_time: Duration::ZERO,
                span_count: 0,
            };
        }

        // Build a map of children self-time sums for parent self-time calculation.
        let duration_map: HashMap<SpanId, Duration> = closed
            .iter()
            .filter_map(|s| s.duration().map(|d| (s.id, d)))
            .collect();

        // For each span, compute self_duration = total - sum(children durations).
        // Children durations are taken from duration_map.
        let self_durations: HashMap<SpanId, Duration> = closed
            .iter()
            .map(|span| {
                let total = duration_map
                    .get(&span.id)
                    .copied()
                    .unwrap_or(Duration::ZERO);
                let children_total: Duration = span
                    .children
                    .iter()
                    .filter_map(|cid| duration_map.get(cid))
                    .copied()
                    .fold(Duration::ZERO, |acc, d| acc + d);
                let self_dur = total.saturating_sub(children_total);
                (span.id, self_dur)
            })
            .collect();

        // Compute overall wall time = max total duration across root spans.
        let root_ids = tracker.root_span_ids();
        let wall_time = root_ids
            .iter()
            .filter_map(|id| duration_map.get(id))
            .copied()
            .fold(Duration::ZERO, |acc, d| acc + d);
        // Fall back to max span duration if no roots (all spans are orphans).
        let wall_time = if wall_time.is_zero() {
            duration_map
                .values()
                .copied()
                .max()
                .unwrap_or(Duration::ZERO)
        } else {
            wall_time
        };

        // Compute nesting depths.
        let depths = self.compute_depths(&closed, &all_spans);

        // Build candidate list and filter by min_self_time.
        let mut candidates: Vec<SpanBottleneck> = closed
            .iter()
            .filter_map(|span| {
                let self_dur = self_durations.get(&span.id).copied()?;
                if self_dur < self.config.min_self_time {
                    return None;
                }
                let total_dur = duration_map
                    .get(&span.id)
                    .copied()
                    .unwrap_or(Duration::ZERO);
                let depth = depths.get(&span.id).copied().unwrap_or(0);
                let fraction = if wall_time.is_zero() {
                    0.0
                } else {
                    self_dur.as_secs_f64() / wall_time.as_secs_f64()
                };

                let kind = self.classify_span(span, self_dur, total_dur, depth);

                Some(SpanBottleneck {
                    span_id: span.id,
                    name: span.name.clone(),
                    total_duration: total_dur,
                    self_duration: self_dur,
                    child_count: span.children.len(),
                    depth,
                    kind,
                    fraction_of_total: fraction,
                })
            })
            .collect();

        // Sort by self-time descending.
        candidates.sort_by(|a, b| b.self_duration.cmp(&a.self_duration));
        candidates.truncate(top_n);

        SpanBottleneckReport {
            bottlenecks: candidates,
            wall_time,
            span_count: closed.len(),
        }
    }

    /// Assigns a `SpanBottleneckKind` to a single span.
    fn classify_span(
        &self,
        span: &Span,
        self_dur: Duration,
        total_dur: Duration,
        depth: usize,
    ) -> SpanBottleneckKind {
        let name_lower = span.name.to_lowercase();

        // I/O hint check.
        let is_io_hint = self
            .config
            .io_name_hints
            .iter()
            .any(|hint| name_lower.contains(hint.as_str()));
        if is_io_hint {
            return SpanBottleneckKind::IoWait;
        }

        // Aggregator: most time is in children.
        if !total_dur.is_zero() {
            let child_frac = 1.0 - (self_dur.as_secs_f64() / total_dur.as_secs_f64());
            if child_frac >= self.config.aggregator_child_fraction {
                return SpanBottleneckKind::Aggregator;
            }
        }

        // Leaf hotspot: no children.
        if span.children.is_empty() {
            if depth > 3 {
                return SpanBottleneckKind::LeafHotspot;
            }
            return SpanBottleneckKind::CpuHotspot;
        }

        // Shallow span with significant self-time.
        if depth <= 1 {
            return SpanBottleneckKind::CpuHotspot;
        }

        SpanBottleneckKind::SelfTimeHeavy
    }

    /// Computes the nesting depth of every span (root = 0).
    fn compute_depths<'a>(
        &self,
        closed: &[&'a Span],
        all_spans: &'a [Span],
    ) -> HashMap<SpanId, usize> {
        let span_map: HashMap<SpanId, &Span> = all_spans.iter().map(|s| (s.id, s)).collect();

        let mut depths: HashMap<SpanId, usize> = HashMap::new();

        for span in closed {
            let depth = Self::compute_depth(span.id, &span_map);
            depths.insert(span.id, depth);
        }
        depths
    }

    /// Walks up the parent chain to compute depth recursively.
    fn compute_depth(id: SpanId, map: &HashMap<SpanId, &Span>) -> usize {
        let mut depth = 0;
        let mut current_id = id;
        // Guard against pathological cycles (should never happen but be safe).
        for _ in 0..1024 {
            match map.get(&current_id).and_then(|s| s.parent_id) {
                Some(pid) => {
                    depth += 1;
                    current_id = pid;
                }
                None => break,
            }
        }
        depth
    }
}

impl Default for SpanBottleneckClassifier {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helper: classify spans from a slice
// ---------------------------------------------------------------------------

/// Convenience function to classify spans without constructing a full `SpanTracker`.
///
/// This is useful when spans have been collected externally (e.g. from
/// serialised data).
#[must_use]
pub fn classify_spans(spans: &[Span], top_n: usize) -> SpanBottleneckReport {
    // Build a minimal SpanTracker by re-using classification logic directly.
    let classifier = SpanBottleneckClassifier::new();

    let closed: Vec<&Span> = spans.iter().filter(|s| s.is_closed()).collect();

    if closed.is_empty() {
        return SpanBottleneckReport {
            bottlenecks: Vec::new(),
            wall_time: Duration::ZERO,
            span_count: 0,
        };
    }

    let duration_map: HashMap<SpanId, Duration> = closed
        .iter()
        .filter_map(|s| s.duration().map(|d| (s.id, d)))
        .collect();

    let self_durations: HashMap<SpanId, Duration> = closed
        .iter()
        .map(|span| {
            let total = duration_map
                .get(&span.id)
                .copied()
                .unwrap_or(Duration::ZERO);
            let children_total: Duration = span
                .children
                .iter()
                .filter_map(|cid| duration_map.get(cid))
                .copied()
                .fold(Duration::ZERO, |acc, d| acc + d);
            (span.id, total.saturating_sub(children_total))
        })
        .collect();

    let span_map: HashMap<SpanId, &Span> = spans.iter().map(|s| (s.id, s)).collect();

    // Wall time = max duration of root spans.
    let wall_time = closed
        .iter()
        .filter(|s| s.parent_id.is_none())
        .filter_map(|s| duration_map.get(&s.id))
        .copied()
        .fold(Duration::ZERO, |acc, d| acc + d);
    let wall_time = if wall_time.is_zero() {
        duration_map
            .values()
            .copied()
            .max()
            .unwrap_or(Duration::ZERO)
    } else {
        wall_time
    };

    let mut candidates: Vec<SpanBottleneck> = closed
        .iter()
        .filter_map(|span| {
            let self_dur = self_durations.get(&span.id).copied()?;
            if self_dur < Duration::from_micros(1) {
                return None;
            }
            let total_dur = duration_map
                .get(&span.id)
                .copied()
                .unwrap_or(Duration::ZERO);
            let depth = SpanBottleneckClassifier::compute_depth(span.id, &span_map);
            let fraction = if wall_time.is_zero() {
                0.0
            } else {
                self_dur.as_secs_f64() / wall_time.as_secs_f64()
            };
            let kind = classifier.classify_span(span, self_dur, total_dur, depth);
            Some(SpanBottleneck {
                span_id: span.id,
                name: span.name.clone(),
                total_duration: total_dur,
                self_duration: self_dur,
                child_count: span.children.len(),
                depth,
                kind,
                fraction_of_total: fraction,
            })
        })
        .collect();

    candidates.sort_by(|a, b| b.self_duration.cmp(&a.self_duration));
    candidates.truncate(top_n);

    SpanBottleneckReport {
        bottlenecks: candidates,
        wall_time,
        span_count: closed.len(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::span::SpanTracker;
    use std::thread;

    #[test]
    fn test_empty_tracker_returns_empty_report() {
        let tracker = SpanTracker::new();
        let classifier = SpanBottleneckClassifier::new();
        let report = classifier.classify(&tracker, 10);
        assert!(report.bottlenecks.is_empty());
        assert_eq!(report.span_count, 0);
    }

    #[test]
    fn test_single_span_is_detected() {
        let tracker = SpanTracker::new();
        {
            let _g = tracker.enter("compute");
            thread::sleep(Duration::from_millis(10));
        }
        let classifier = SpanBottleneckClassifier::new();
        let report = classifier.classify(&tracker, 10);
        assert_eq!(report.span_count, 1);
        assert!(!report.bottlenecks.is_empty());
        assert_eq!(report.bottlenecks[0].name, "compute");
    }

    #[test]
    fn test_top_n_limits_results() {
        let tracker = SpanTracker::new();
        for i in 0..10 {
            let _g = tracker.enter(format!("span_{}", i));
            // Each span gets slightly different timing due to loop overhead.
            drop(_g);
        }
        let classifier = SpanBottleneckClassifier::new();
        let report = classifier.classify(&tracker, 3);
        assert!(report.bottlenecks.len() <= 3);
    }

    #[test]
    fn test_io_hint_classification() {
        let tracker = SpanTracker::new();
        {
            let _g = tracker.enter("file_read");
            thread::sleep(Duration::from_millis(5));
        }
        let classifier = SpanBottleneckClassifier::new();
        let report = classifier.classify(&tracker, 10);
        let b = &report.bottlenecks[0];
        assert_eq!(b.kind, SpanBottleneckKind::IoWait);
    }

    #[test]
    fn test_aggregator_classification() {
        let tracker = SpanTracker::new();
        {
            let _outer = tracker.enter("pipeline");
            {
                let _inner = tracker.enter("heavy_work");
                thread::sleep(Duration::from_millis(20));
            }
        }
        let classifier = SpanBottleneckClassifier::new();
        let report = classifier.classify(&tracker, 10);
        // pipeline should be classified as aggregator since heavy_work takes most time.
        let pipeline = report.bottlenecks.iter().find(|b| b.name == "pipeline");
        if let Some(p) = pipeline {
            assert_eq!(p.kind, SpanBottleneckKind::Aggregator);
        }
    }

    #[test]
    fn test_leaf_hotspot_classification() {
        let tracker = SpanTracker::new();
        {
            let _l0 = tracker.enter("root");
            {
                let _l1 = tracker.enter("level1");
                {
                    let _l2 = tracker.enter("level2");
                    {
                        let _l3 = tracker.enter("level3");
                        {
                            let _l4 = tracker.enter("leaf_compute");
                            thread::sleep(Duration::from_millis(10));
                        }
                    }
                }
            }
        }
        let classifier = SpanBottleneckClassifier::new();
        let report = classifier.classify(&tracker, 10);
        let leaf = report.bottlenecks.iter().find(|b| b.name == "leaf_compute");
        if let Some(l) = leaf {
            assert_eq!(l.kind, SpanBottleneckKind::LeafHotspot);
        }
    }

    #[test]
    fn test_primary_is_highest_self_time() {
        let tracker = SpanTracker::new();
        {
            let _slow = tracker.enter("slow_fn");
            thread::sleep(Duration::from_millis(20));
        }
        {
            let _fast = tracker.enter("fast_fn");
            thread::sleep(Duration::from_millis(2));
        }
        let classifier = SpanBottleneckClassifier::new();
        let report = classifier.classify(&tracker, 10);
        let primary = report.primary().expect("should have primary");
        assert_eq!(primary.name, "slow_fn");
    }

    #[test]
    fn test_report_summary_non_empty() {
        let tracker = SpanTracker::new();
        {
            let _g = tracker.enter("work");
            thread::sleep(Duration::from_millis(5));
        }
        let classifier = SpanBottleneckClassifier::new();
        let report = classifier.classify(&tracker, 10);
        let summary = report.summary();
        assert!(summary.contains("Span Bottleneck Report"));
        assert!(summary.contains("work"));
    }

    #[test]
    fn test_describe_contains_name() {
        let tracker = SpanTracker::new();
        {
            let _g = tracker.enter("my_function");
            thread::sleep(Duration::from_millis(5));
        }
        let classifier = SpanBottleneckClassifier::new();
        let report = classifier.classify(&tracker, 10);
        let desc = report.bottlenecks[0].describe();
        assert!(desc.contains("my_function"));
    }

    #[test]
    fn test_fraction_is_positive() {
        let tracker = SpanTracker::new();
        {
            let _g = tracker.enter("span");
            thread::sleep(Duration::from_millis(5));
        }
        let classifier = SpanBottleneckClassifier::new();
        let report = classifier.classify(&tracker, 10);
        assert!(report.bottlenecks[0].fraction_of_total > 0.0);
    }

    #[test]
    fn test_wall_time_covers_root_duration() {
        let tracker = SpanTracker::new();
        {
            let _g = tracker.enter("root");
            thread::sleep(Duration::from_millis(10));
        }
        let classifier = SpanBottleneckClassifier::new();
        let report = classifier.classify(&tracker, 10);
        assert!(report.wall_time >= Duration::from_millis(10));
    }

    #[test]
    fn test_classify_spans_helper() {
        let tracker = SpanTracker::new();
        {
            let _g = tracker.enter("fn_a");
            thread::sleep(Duration::from_millis(5));
        }
        let spans = tracker.all_spans();
        let report = classify_spans(&spans, 10);
        assert_eq!(report.span_count, 1);
        assert!(!report.bottlenecks.is_empty());
    }

    #[test]
    fn test_kind_labels() {
        assert_eq!(SpanBottleneckKind::CpuHotspot.label(), "CPU hotspot");
        assert_eq!(SpanBottleneckKind::IoWait.label(), "I/O wait");
        assert_eq!(
            SpanBottleneckKind::Aggregator.label(),
            "Aggregator (high child overhead)"
        );
        assert_eq!(SpanBottleneckKind::SelfTimeHeavy.label(), "Self-time heavy");
        assert_eq!(SpanBottleneckKind::LeafHotspot.label(), "Leaf hotspot");
    }
}
