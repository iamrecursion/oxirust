#![allow(dead_code)]

//! Segment-level summary statistics aggregation.
//!
//! This module divides a video/audio stream into fixed-duration segments and
//! computes aggregate quality, complexity, and content metrics for each segment.
//! This is useful for heatmap visualization, quality dashboards, and identifying
//! problematic regions in long-form content.

use std::collections::BTreeMap;

/// Duration of a segment in frames.
const DEFAULT_SEGMENT_FRAMES: u64 = 300;

/// A metric value for a single frame within a segment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameMetric {
    /// Frame index.
    pub frame_index: u64,
    /// Metric value.
    pub value: f64,
}

impl FrameMetric {
    /// Create a new frame metric.
    pub fn new(frame_index: u64, value: f64) -> Self {
        Self { frame_index, value }
    }
}

/// Aggregate statistics for a single segment.
#[derive(Debug, Clone, PartialEq)]
pub struct SegmentStats {
    /// Segment index (0-based).
    pub segment_index: u64,
    /// First frame index in this segment.
    pub start_frame: u64,
    /// Last frame index in this segment (inclusive).
    pub end_frame: u64,
    /// Number of frames in this segment.
    pub frame_count: usize,
    /// Mean value of the metric in this segment.
    pub mean: f64,
    /// Minimum value.
    pub min: f64,
    /// Maximum value.
    pub max: f64,
    /// Standard deviation.
    pub std_dev: f64,
    /// Median value.
    pub median: f64,
}

/// Overall summary across all segments.
#[derive(Debug, Clone, PartialEq)]
pub struct OverallSummary {
    /// Total number of segments.
    pub segment_count: usize,
    /// Total number of frames.
    pub total_frames: usize,
    /// Global mean across all frames.
    pub global_mean: f64,
    /// Global min across all frames.
    pub global_min: f64,
    /// Global max across all frames.
    pub global_max: f64,
    /// Segment with the highest mean value.
    pub worst_segment_index: Option<u64>,
    /// Segment with the lowest mean value.
    pub best_segment_index: Option<u64>,
}

/// Quality level classification for a segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentQuality {
    /// Excellent quality.
    Excellent,
    /// Good quality.
    Good,
    /// Fair quality.
    Fair,
    /// Poor quality.
    Poor,
    /// Bad quality.
    Bad,
}

impl SegmentQuality {
    /// Classify from a normalized score (0.0 = worst, 1.0 = best).
    pub fn from_score(score: f64) -> Self {
        if score >= 0.9 {
            Self::Excellent
        } else if score >= 0.7 {
            Self::Good
        } else if score >= 0.5 {
            Self::Fair
        } else if score >= 0.3 {
            Self::Poor
        } else {
            Self::Bad
        }
    }

    /// Returns a human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            Self::Excellent => "excellent",
            Self::Good => "good",
            Self::Fair => "fair",
            Self::Poor => "poor",
            Self::Bad => "bad",
        }
    }
}

/// Accumulator for segment-level summary statistics.
#[derive(Debug)]
pub struct SegmentSummarizer {
    /// Number of frames per segment.
    segment_duration_frames: u64,
    /// All frame metrics, grouped by segment index.
    segments: BTreeMap<u64, Vec<f64>>,
    /// Total frames added.
    total_frames: usize,
}

impl SegmentSummarizer {
    /// Create a new summarizer with the given segment duration in frames.
    pub fn new(segment_duration_frames: u64) -> Self {
        let dur = if segment_duration_frames == 0 {
            DEFAULT_SEGMENT_FRAMES
        } else {
            segment_duration_frames
        };
        Self {
            segment_duration_frames: dur,
            segments: BTreeMap::new(),
            total_frames: 0,
        }
    }

    /// Create with default segment duration (300 frames / ~10s at 30fps).
    pub fn with_defaults() -> Self {
        Self::new(DEFAULT_SEGMENT_FRAMES)
    }

    /// Add a frame metric value.
    pub fn add_metric(&mut self, metric: FrameMetric) {
        let seg_idx = metric.frame_index / self.segment_duration_frames;
        self.segments.entry(seg_idx).or_default().push(metric.value);
        self.total_frames += 1;
    }

    /// Add a batch of frame metrics.
    pub fn add_metrics(&mut self, metrics: &[FrameMetric]) {
        for m in metrics {
            self.add_metric(*m);
        }
    }

    /// Compute statistics for a single segment.
    #[allow(clippy::cast_precision_loss)]
    fn compute_segment_stats(seg_idx: u64, values: &[f64], seg_duration: u64) -> SegmentStats {
        let n = values.len();
        let start_frame = seg_idx * seg_duration;
        let end_frame = start_frame + (n as u64).saturating_sub(1);

        if n == 0 {
            return SegmentStats {
                segment_index: seg_idx,
                start_frame,
                end_frame,
                frame_count: 0,
                mean: 0.0,
                min: 0.0,
                max: 0.0,
                std_dev: 0.0,
                median: 0.0,
            };
        }

        let sum: f64 = values.iter().sum();
        let mean = sum / n as f64;
        let min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        let variance = if n > 1 {
            values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64
        } else {
            0.0
        };

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = if n % 2 == 0 {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        };

        SegmentStats {
            segment_index: seg_idx,
            start_frame,
            end_frame,
            frame_count: n,
            mean,
            min,
            max,
            std_dev: variance.sqrt(),
            median,
        }
    }

    /// Compute stats for all segments.
    pub fn compute_all_stats(&self) -> Vec<SegmentStats> {
        self.segments
            .iter()
            .map(|(&idx, values)| {
                Self::compute_segment_stats(idx, values, self.segment_duration_frames)
            })
            .collect()
    }

    /// Compute an overall summary across all segments.
    #[allow(clippy::cast_precision_loss)]
    pub fn compute_overall(&self) -> OverallSummary {
        let all_stats = self.compute_all_stats();
        if all_stats.is_empty() {
            return OverallSummary {
                segment_count: 0,
                total_frames: 0,
                global_mean: 0.0,
                global_min: 0.0,
                global_max: 0.0,
                worst_segment_index: None,
                best_segment_index: None,
            };
        }

        let all_values: Vec<f64> = self.segments.values().flatten().copied().collect();
        let sum: f64 = all_values.iter().sum();
        let global_mean = if all_values.is_empty() {
            0.0
        } else {
            sum / all_values.len() as f64
        };
        let global_min = all_values.iter().copied().fold(f64::INFINITY, f64::min);
        let global_max = all_values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        let worst = all_stats
            .iter()
            .max_by(|a, b| {
                a.mean
                    .partial_cmp(&b.mean)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|s| s.segment_index);

        let best = all_stats
            .iter()
            .min_by(|a, b| {
                a.mean
                    .partial_cmp(&b.mean)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|s| s.segment_index);

        OverallSummary {
            segment_count: all_stats.len(),
            total_frames: self.total_frames,
            global_mean,
            global_min,
            global_max,
            worst_segment_index: worst,
            best_segment_index: best,
        }
    }

    /// Return the number of segments.
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Return the total number of frames added.
    pub fn total_frames(&self) -> usize {
        self.total_frames
    }

    /// Clear all data.
    pub fn clear(&mut self) {
        self.segments.clear();
        self.total_frames = 0;
    }
}

impl Default for SegmentSummarizer {
    fn default() -> Self {
        Self::with_defaults()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_metric_creation() {
        let m = FrameMetric::new(42, 0.95);
        assert_eq!(m.frame_index, 42);
        assert!((m.value - 0.95).abs() < f64::EPSILON);
    }

    #[test]
    fn test_empty_summarizer() {
        let s = SegmentSummarizer::with_defaults();
        assert_eq!(s.segment_count(), 0);
        assert_eq!(s.total_frames(), 0);
    }

    #[test]
    fn test_single_segment() {
        let mut s = SegmentSummarizer::new(10);
        for i in 0..10 {
            s.add_metric(FrameMetric::new(i, 0.5));
        }
        assert_eq!(s.segment_count(), 1);
        let stats = s.compute_all_stats();
        assert_eq!(stats.len(), 1);
        assert!((stats[0].mean - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_multiple_segments() {
        let mut s = SegmentSummarizer::new(5);
        // Segment 0: frames 0-4
        for i in 0..5 {
            s.add_metric(FrameMetric::new(i, 1.0));
        }
        // Segment 1: frames 5-9
        for i in 5..10 {
            s.add_metric(FrameMetric::new(i, 2.0));
        }
        assert_eq!(s.segment_count(), 2);
        let stats = s.compute_all_stats();
        assert!((stats[0].mean - 1.0).abs() < 0.001);
        assert!((stats[1].mean - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_overall_summary() {
        let mut s = SegmentSummarizer::new(5);
        for i in 0..5 {
            s.add_metric(FrameMetric::new(i, 1.0));
        }
        for i in 5..10 {
            s.add_metric(FrameMetric::new(i, 3.0));
        }
        let overall = s.compute_overall();
        assert_eq!(overall.segment_count, 2);
        assert_eq!(overall.total_frames, 10);
        assert!((overall.global_mean - 2.0).abs() < 0.001);
        assert!((overall.global_min - 1.0).abs() < 0.001);
        assert!((overall.global_max - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_worst_best_segment() {
        let mut s = SegmentSummarizer::new(5);
        for i in 0..5 {
            s.add_metric(FrameMetric::new(i, 0.2));
        }
        for i in 5..10 {
            s.add_metric(FrameMetric::new(i, 0.8));
        }
        let overall = s.compute_overall();
        assert_eq!(overall.best_segment_index, Some(0));
        assert_eq!(overall.worst_segment_index, Some(1));
    }

    #[test]
    fn test_segment_stats_std_dev() {
        let mut s = SegmentSummarizer::new(100);
        // All same value => std_dev ~= 0
        for i in 0..50 {
            s.add_metric(FrameMetric::new(i, 5.0));
        }
        let stats = s.compute_all_stats();
        assert!(stats[0].std_dev < 0.001);
    }

    #[test]
    fn test_segment_stats_median_odd() {
        let mut s = SegmentSummarizer::new(100);
        s.add_metric(FrameMetric::new(0, 1.0));
        s.add_metric(FrameMetric::new(1, 3.0));
        s.add_metric(FrameMetric::new(2, 5.0));
        let stats = s.compute_all_stats();
        assert!((stats[0].median - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_segment_stats_median_even() {
        let mut s = SegmentSummarizer::new(100);
        s.add_metric(FrameMetric::new(0, 1.0));
        s.add_metric(FrameMetric::new(1, 3.0));
        s.add_metric(FrameMetric::new(2, 5.0));
        s.add_metric(FrameMetric::new(3, 7.0));
        let stats = s.compute_all_stats();
        assert!((stats[0].median - 4.0).abs() < 0.001);
    }

    #[test]
    fn test_segment_quality_classification() {
        assert_eq!(SegmentQuality::from_score(0.95), SegmentQuality::Excellent);
        assert_eq!(SegmentQuality::from_score(0.75), SegmentQuality::Good);
        assert_eq!(SegmentQuality::from_score(0.55), SegmentQuality::Fair);
        assert_eq!(SegmentQuality::from_score(0.35), SegmentQuality::Poor);
        assert_eq!(SegmentQuality::from_score(0.1), SegmentQuality::Bad);
    }

    #[test]
    fn test_quality_labels() {
        assert_eq!(SegmentQuality::Excellent.label(), "excellent");
        assert_eq!(SegmentQuality::Bad.label(), "bad");
    }

    #[test]
    fn test_add_metrics_batch() {
        let mut s = SegmentSummarizer::new(10);
        let metrics: Vec<FrameMetric> = (0..20)
            .map(|i| FrameMetric::new(i, i as f64 * 0.1))
            .collect();
        s.add_metrics(&metrics);
        assert_eq!(s.total_frames(), 20);
        assert_eq!(s.segment_count(), 2);
    }

    #[test]
    fn test_clear() {
        let mut s = SegmentSummarizer::new(10);
        s.add_metric(FrameMetric::new(0, 1.0));
        s.clear();
        assert_eq!(s.segment_count(), 0);
        assert_eq!(s.total_frames(), 0);
    }
}
