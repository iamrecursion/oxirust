//! Shot statistics and summary reporting for `oximedia-shots`.
//!
//! Aggregates per-shot data into a [`ShotReport`] that summarises durations,
//! type distributions, and other editorial metrics useful for QC and review.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Per-shot statistics ───────────────────────────────────────────────────────

/// Statistics captured for a single shot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShotStats {
    /// Unique shot identifier (e.g. sequential index in the edit).
    pub shot_id: u64,
    /// Duration of the shot expressed as a frame count.
    pub duration_frames: u32,
    /// Human-readable shot-size label (e.g. `"WS"`, `"CU"`).
    pub size_label: String,
    /// Human-readable camera-angle label (e.g. `"Eye Level"`).
    pub angle_label: String,
    /// Classifier confidence score in `[0.0, 1.0]`.
    pub confidence: f32,
    /// Whether the shot was flagged as a continuity issue.
    pub has_continuity_issue: bool,
}

impl ShotStats {
    /// Creates a new [`ShotStats`] entry.
    #[must_use]
    pub fn new(
        shot_id: u64,
        duration_frames: u32,
        size_label: impl Into<String>,
        angle_label: impl Into<String>,
        confidence: f32,
    ) -> Self {
        Self {
            shot_id,
            duration_frames,
            size_label: size_label.into(),
            angle_label: angle_label.into(),
            confidence: confidence.clamp(0.0, 1.0),
            has_continuity_issue: false,
        }
    }
}

// ── Report ───────────────────────────────────────────────────────────────────

/// Aggregated report over a collection of [`ShotStats`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShotReport {
    /// Title or identifier for the source material (e.g. clip name).
    pub title: String,
    /// Frame rate used when converting frame counts to seconds.
    pub frame_rate: f32,
    /// Individual shot records in editorial order.
    pub shots: Vec<ShotStats>,
}

impl ShotReport {
    /// Creates a new empty report.
    #[must_use]
    pub fn new(title: impl Into<String>, frame_rate: f32) -> Self {
        Self {
            title: title.into(),
            frame_rate: frame_rate.max(1.0),
            shots: Vec::new(),
        }
    }

    /// Appends a [`ShotStats`] record to this report.
    pub fn push(&mut self, stats: ShotStats) {
        self.shots.push(stats);
    }

    /// Total number of shots in the report.
    #[must_use]
    pub fn shot_count(&self) -> usize {
        self.shots.len()
    }

    /// Returns the average shot duration expressed in frames, or `0.0` when
    /// the report contains no shots.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn avg_duration_frames(&self) -> f64 {
        if self.shots.is_empty() {
            return 0.0;
        }
        let total: u64 = self
            .shots
            .iter()
            .map(|s| u64::from(s.duration_frames))
            .sum();
        total as f64 / self.shots.len() as f64
    }

    /// Returns the average shot duration in seconds.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn avg_duration_seconds(&self) -> f64 {
        self.avg_duration_frames() / self.frame_rate as f64
    }

    /// Returns the shortest shot duration in frames, or `None` for empty reports.
    #[must_use]
    pub fn min_duration_frames(&self) -> Option<u32> {
        self.shots.iter().map(|s| s.duration_frames).min()
    }

    /// Returns the longest shot duration in frames, or `None` for empty reports.
    #[must_use]
    pub fn max_duration_frames(&self) -> Option<u32> {
        self.shots.iter().map(|s| s.duration_frames).max()
    }

    /// Counts occurrences of each `size_label` in the report.
    #[must_use]
    pub fn size_distribution(&self) -> HashMap<String, usize> {
        let mut dist: HashMap<String, usize> = HashMap::new();
        for s in &self.shots {
            *dist.entry(s.size_label.clone()).or_insert(0) += 1;
        }
        dist
    }

    /// Returns the number of shots flagged with continuity issues.
    #[must_use]
    pub fn continuity_issue_count(&self) -> usize {
        self.shots.iter().filter(|s| s.has_continuity_issue).count()
    }

    /// Mean confidence across all shots, or `0.0` for empty reports.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn mean_confidence(&self) -> f32 {
        if self.shots.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.shots.iter().map(|s| s.confidence).sum();
        sum / self.shots.len() as f32
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_report() -> ShotReport {
        let mut r = ShotReport::new("clip_001", 24.0);
        r.push(ShotStats::new(0, 48, "WS", "Eye Level", 0.9));
        r.push(ShotStats::new(1, 24, "CU", "High Angle", 0.7));
        r.push(ShotStats::new(2, 72, "MS", "Eye Level", 0.8));
        r
    }

    #[test]
    fn test_report_new_empty() {
        let r = ShotReport::new("test", 25.0);
        assert_eq!(r.shot_count(), 0);
    }

    #[test]
    fn test_push_increments_count() {
        let mut r = ShotReport::new("test", 25.0);
        r.push(ShotStats::new(0, 50, "WS", "Eye Level", 0.8));
        assert_eq!(r.shot_count(), 1);
    }

    #[test]
    fn test_avg_duration_frames_empty() {
        let r = ShotReport::new("empty", 25.0);
        assert!((r.avg_duration_frames() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_avg_duration_frames() {
        let r = make_report();
        // (48 + 24 + 72) / 3 = 48.0
        assert!((r.avg_duration_frames() - 48.0).abs() < 1e-9);
    }

    #[test]
    fn test_avg_duration_seconds() {
        let r = make_report(); // 24 fps
                               // 48 frames / 24 fps = 2.0 seconds
        assert!((r.avg_duration_seconds() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_min_duration_frames() {
        let r = make_report();
        assert_eq!(r.min_duration_frames(), Some(24));
    }

    #[test]
    fn test_max_duration_frames() {
        let r = make_report();
        assert_eq!(r.max_duration_frames(), Some(72));
    }

    #[test]
    fn test_min_max_empty() {
        let r = ShotReport::new("empty", 25.0);
        assert!(r.min_duration_frames().is_none());
        assert!(r.max_duration_frames().is_none());
    }

    #[test]
    fn test_size_distribution() {
        let r = make_report();
        let dist = r.size_distribution();
        assert_eq!(dist["WS"], 1);
        assert_eq!(dist["CU"], 1);
        assert_eq!(dist["MS"], 1);
    }

    #[test]
    fn test_continuity_issue_count_zero() {
        let r = make_report();
        assert_eq!(r.continuity_issue_count(), 0);
    }

    #[test]
    fn test_continuity_issue_count_some() {
        let mut r = make_report();
        r.shots[1].has_continuity_issue = true;
        assert_eq!(r.continuity_issue_count(), 1);
    }

    #[test]
    fn test_mean_confidence() {
        let r = make_report(); // 0.9 + 0.7 + 0.8 = 2.4 / 3
        assert!((r.mean_confidence() - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_mean_confidence_empty() {
        let r = ShotReport::new("empty", 25.0);
        assert!((r.mean_confidence() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_confidence_clamped() {
        let s = ShotStats::new(0, 10, "WS", "Eye Level", 2.0);
        assert!((s.confidence - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_frame_rate_floor() {
        let r = ShotReport::new("title", -5.0);
        assert!((r.frame_rate - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_single_shot_report() {
        let mut r = ShotReport::new("single", 30.0);
        r.push(ShotStats::new(0, 90, "MS", "Eye Level", 1.0));
        assert!((r.avg_duration_frames() - 90.0).abs() < 1e-9);
        assert!((r.avg_duration_seconds() - 3.0).abs() < 1e-9);
    }
}
