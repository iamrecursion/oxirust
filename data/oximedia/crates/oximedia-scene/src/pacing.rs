//! Scene pacing analysis.
//!
//! Analyses the editing rhythm of a video sequence by examining the distribution
//! and patterns of shot durations.

/// Aggregate metrics describing the pacing of a sequence of shots.
#[derive(Debug, Clone)]
pub struct PacingMetrics {
    /// Average shot duration in frames.
    pub avg_shot_duration_frames: f64,
    /// Total number of shots.
    pub shot_count: usize,
    /// Cuts per minute, computed from the average shot duration and FPS.
    pub cut_rate_per_minute: f64,
    /// Duration of the shortest shot in frames.
    pub fastest_shot: u64,
    /// Duration of the longest shot in frames.
    pub slowest_shot: u64,
}

impl PacingMetrics {
    /// Compute a normalised pacing score (0.0 = very slow, 1.0 = hyper-active).
    ///
    /// Derived from the cut rate per minute; calibrated so that
    /// < 10 cuts/min ≈ 0.0 and > 120 cuts/min ≈ 1.0.
    #[must_use]
    pub fn pacing_score(&self) -> f64 {
        ((self.cut_rate_per_minute - 10.0) / 110.0).clamp(0.0, 1.0)
    }

    /// Return `true` when the sequence is fast-paced (score > 0.5).
    #[must_use]
    pub fn is_fast_paced(&self) -> bool {
        self.pacing_score() > 0.5
    }
}

/// A qualitative label for the pacing style of a sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PacingStyle {
    /// Slow, contemplative editing (< 20 cuts/min).
    Slow,
    /// Moderate, narrative pacing (20–40 cuts/min).
    Moderate,
    /// Fast editing common in action/thriller (40–80 cuts/min).
    Fast,
    /// Very rapid editing (> 80 cuts/min).
    HyperActive,
}

/// Classify a `PacingMetrics` into a `PacingStyle`.
#[must_use]
pub fn classify_pacing(metrics: &PacingMetrics) -> PacingStyle {
    let cpm = metrics.cut_rate_per_minute;
    if cpm < 20.0 {
        PacingStyle::Slow
    } else if cpm < 40.0 {
        PacingStyle::Moderate
    } else if cpm < 80.0 {
        PacingStyle::Fast
    } else {
        PacingStyle::HyperActive
    }
}

/// An ordered list of shot durations together with the frame rate.
#[derive(Debug, Clone)]
pub struct ShotList {
    /// Individual shot durations in frames.
    durations_frames: Vec<u64>,
    /// Frames per second of the source material.
    fps: f64,
}

impl ShotList {
    /// Create a new, empty `ShotList` for material recorded at `fps`.
    #[must_use]
    pub fn new(fps: f64) -> Self {
        Self {
            durations_frames: Vec::new(),
            fps,
        }
    }

    /// Append a shot with the given duration in frames.
    pub fn add_shot(&mut self, duration: u64) {
        self.durations_frames.push(duration);
    }

    /// Compute aggregate `PacingMetrics` from all stored shots.
    ///
    /// Returns a zeroed-out `PacingMetrics` when no shots have been added.
    #[must_use]
    pub fn compute_metrics(&self) -> PacingMetrics {
        let n = self.durations_frames.len();
        if n == 0 {
            return PacingMetrics {
                avg_shot_duration_frames: 0.0,
                shot_count: 0,
                cut_rate_per_minute: 0.0,
                fastest_shot: 0,
                slowest_shot: 0,
            };
        }

        let total: u64 = self.durations_frames.iter().sum();
        let avg = total as f64 / n as f64;

        let fastest = *self.durations_frames.iter().min().unwrap_or(&0);
        let slowest = *self.durations_frames.iter().max().unwrap_or(&0);

        // Cuts per minute = cuts / total_seconds * 60
        let total_seconds = total as f64 / self.fps.max(f64::EPSILON);
        let cut_rate = if total_seconds > 0.0 {
            n as f64 / total_seconds * 60.0
        } else {
            0.0
        };

        PacingMetrics {
            avg_shot_duration_frames: avg,
            shot_count: n,
            cut_rate_per_minute: cut_rate,
            fastest_shot: fastest,
            slowest_shot: slowest,
        }
    }

    /// Compute the standard deviation of shot durations as a fraction of the mean.
    ///
    /// Returns 0.0 when there are fewer than two shots or the mean is zero.
    #[must_use]
    pub fn rhythm_deviation(&self) -> f64 {
        let n = self.durations_frames.len();
        if n < 2 {
            return 0.0;
        }
        let mean = self.durations_frames.iter().sum::<u64>() as f64 / n as f64;
        if mean < f64::EPSILON {
            return 0.0;
        }
        let variance = self
            .durations_frames
            .iter()
            .map(|&d| {
                let diff = d as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / n as f64;
        variance.sqrt() / mean
    }
}

/// Detect pacing changes using a sliding window over shot durations.
///
/// Returns a `Vec` of `(window_start_index, PacingStyle)` pairs, one per window.
/// `window_size` must be at least 1; if `shots` is shorter than `window_size` a
/// single entry covering the whole slice is returned (or an empty Vec for empty
/// input).
#[must_use]
pub fn detect_pacing_changes(
    shots: &[u64],
    window_size: usize,
    fps: f64,
) -> Vec<(usize, PacingStyle)> {
    if shots.is_empty() || window_size == 0 {
        return Vec::new();
    }

    let step = window_size.max(1);
    let mut results = Vec::new();

    let mut i = 0;
    while i < shots.len() {
        let end = (i + step).min(shots.len());
        let window = &shots[i..end];

        let mut list = ShotList::new(fps);
        for &d in window {
            list.add_shot(d);
        }
        let metrics = list.compute_metrics();
        let style = classify_pacing(&metrics);
        results.push((i, style));

        i += step;
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pacing_metrics_score_slow() {
        let m = PacingMetrics {
            avg_shot_duration_frames: 120.0,
            shot_count: 10,
            cut_rate_per_minute: 5.0,
            fastest_shot: 100,
            slowest_shot: 150,
        };
        assert!((m.pacing_score() - 0.0).abs() < f64::EPSILON);
        assert!(!m.is_fast_paced());
    }

    #[test]
    fn test_pacing_metrics_score_fast() {
        let m = PacingMetrics {
            avg_shot_duration_frames: 10.0,
            shot_count: 100,
            cut_rate_per_minute: 120.0,
            fastest_shot: 5,
            slowest_shot: 20,
        };
        assert!((m.pacing_score() - 1.0).abs() < f64::EPSILON);
        assert!(m.is_fast_paced());
    }

    #[test]
    fn test_pacing_metrics_is_fast_paced_boundary() {
        // score = (65 - 10) / 110 = 55/110 = 0.5, not > 0.5
        let m = PacingMetrics {
            avg_shot_duration_frames: 30.0,
            shot_count: 20,
            cut_rate_per_minute: 65.0,
            fastest_shot: 20,
            slowest_shot: 50,
        };
        assert!(!m.is_fast_paced());
    }

    #[test]
    fn test_classify_pacing_slow() {
        let m = PacingMetrics {
            avg_shot_duration_frames: 200.0,
            shot_count: 5,
            cut_rate_per_minute: 10.0,
            fastest_shot: 180,
            slowest_shot: 220,
        };
        assert_eq!(classify_pacing(&m), PacingStyle::Slow);
    }

    #[test]
    fn test_classify_pacing_moderate() {
        let m = PacingMetrics {
            avg_shot_duration_frames: 80.0,
            shot_count: 10,
            cut_rate_per_minute: 30.0,
            fastest_shot: 60,
            slowest_shot: 100,
        };
        assert_eq!(classify_pacing(&m), PacingStyle::Moderate);
    }

    #[test]
    fn test_classify_pacing_fast() {
        let m = PacingMetrics {
            avg_shot_duration_frames: 30.0,
            shot_count: 30,
            cut_rate_per_minute: 60.0,
            fastest_shot: 20,
            slowest_shot: 50,
        };
        assert_eq!(classify_pacing(&m), PacingStyle::Fast);
    }

    #[test]
    fn test_classify_pacing_hyperactive() {
        let m = PacingMetrics {
            avg_shot_duration_frames: 10.0,
            shot_count: 100,
            cut_rate_per_minute: 100.0,
            fastest_shot: 5,
            slowest_shot: 20,
        };
        assert_eq!(classify_pacing(&m), PacingStyle::HyperActive);
    }

    #[test]
    fn test_shot_list_empty_metrics() {
        let list = ShotList::new(24.0);
        let m = list.compute_metrics();
        assert_eq!(m.shot_count, 0);
        assert!((m.avg_shot_duration_frames - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_shot_list_metrics_basic() {
        let mut list = ShotList::new(24.0);
        list.add_shot(48); // 2 seconds
        list.add_shot(24); // 1 second
        let m = list.compute_metrics();
        assert_eq!(m.shot_count, 2);
        assert!((m.avg_shot_duration_frames - 36.0).abs() < f64::EPSILON);
        assert_eq!(m.fastest_shot, 24);
        assert_eq!(m.slowest_shot, 48);
    }

    #[test]
    fn test_shot_list_cut_rate() {
        // 60 shots each 1 second at 24 fps → 60 shots in 60 seconds → 60 cuts/min
        let mut list = ShotList::new(24.0);
        for _ in 0..60 {
            list.add_shot(24);
        }
        let m = list.compute_metrics();
        assert!((m.cut_rate_per_minute - 60.0).abs() < 0.01);
    }

    #[test]
    fn test_rhythm_deviation_uniform() {
        let mut list = ShotList::new(24.0);
        for _ in 0..5 {
            list.add_shot(48);
        }
        assert!((list.rhythm_deviation() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_rhythm_deviation_nonzero() {
        let mut list = ShotList::new(24.0);
        list.add_shot(24);
        list.add_shot(96);
        assert!(list.rhythm_deviation() > 0.0);
    }

    #[test]
    fn test_detect_pacing_changes_empty() {
        let result = detect_pacing_changes(&[], 3, 24.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_detect_pacing_changes_single_window() {
        let shots = vec![48u64; 10];
        let result = detect_pacing_changes(&shots, 10, 24.0);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, 0);
    }

    #[test]
    fn test_detect_pacing_changes_multiple_windows() {
        let shots = vec![48u64; 6];
        let result = detect_pacing_changes(&shots, 3, 24.0);
        // Two windows: [0..3) and [3..6)
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].0, 0);
        assert_eq!(result[1].0, 3);
    }
}
