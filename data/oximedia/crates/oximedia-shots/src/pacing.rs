//! Shot pacing and editing rhythm analysis for `oximedia-shots`.
//!
//! Provides [`PacingLevel`] classification, [`PacingProfile`] aggregation,
//! and a [`PacingAnalyzer`] that computes Average Shot Length (ASL) and
//! related rhythm metrics commonly used in film editing analysis.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Pacing level
// ---------------------------------------------------------------------------

/// Qualitative pacing classification derived from shot duration statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PacingLevel {
    /// Very fast editing, typical of action or montage sequences.
    Rapid,
    /// Brisk editing, typical of dialogue scenes.
    Fast,
    /// Moderate editing pace.
    Moderate,
    /// Slow, contemplative editing.
    Slow,
    /// Very long takes, often associated with arthouse or oner sequences.
    Lingering,
}

impl PacingLevel {
    /// Returns a human-readable label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Rapid => "Rapid",
            Self::Fast => "Fast",
            Self::Moderate => "Moderate",
            Self::Slow => "Slow",
            Self::Lingering => "Lingering",
        }
    }

    /// Classifies an ASL (Average Shot Length) in seconds into a pacing level.
    ///
    /// Thresholds are approximate industry conventions:
    /// * < 2 s  Rapid
    /// * 2..4 s  Fast
    /// * 4..8 s  Moderate
    /// * 8..15 s  Slow
    /// * >= 15 s  Lingering
    #[must_use]
    pub fn from_asl_seconds(asl: f64) -> Self {
        if asl < 2.0 {
            Self::Rapid
        } else if asl < 4.0 {
            Self::Fast
        } else if asl < 8.0 {
            Self::Moderate
        } else if asl < 15.0 {
            Self::Slow
        } else {
            Self::Lingering
        }
    }

    /// Returns all variants.
    #[must_use]
    pub const fn all() -> &'static [PacingLevel] {
        &[
            Self::Rapid,
            Self::Fast,
            Self::Moderate,
            Self::Slow,
            Self::Lingering,
        ]
    }
}

impl std::fmt::Display for PacingLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

// ---------------------------------------------------------------------------
// Pacing profile
// ---------------------------------------------------------------------------

/// Aggregated pacing metrics for a collection of shots.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PacingProfile {
    /// Number of shots analysed.
    pub shot_count: usize,
    /// Average shot length in frames.
    pub asl_frames: f64,
    /// Average shot length in seconds (using provided frame-rate).
    pub asl_seconds: f64,
    /// Median shot length in frames.
    pub median_frames: f64,
    /// Standard deviation of shot lengths in frames.
    pub std_dev_frames: f64,
    /// Shortest shot in frames.
    pub min_frames: u64,
    /// Longest shot in frames.
    pub max_frames: u64,
    /// Derived qualitative pacing level.
    pub level: PacingLevel,
    /// Cutting-rate (cuts per minute).
    pub cuts_per_minute: f64,
}

impl Default for PacingProfile {
    fn default() -> Self {
        Self {
            shot_count: 0,
            asl_frames: 0.0,
            asl_seconds: 0.0,
            median_frames: 0.0,
            std_dev_frames: 0.0,
            min_frames: 0,
            max_frames: 0,
            level: PacingLevel::Moderate,
            cuts_per_minute: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Pacing analyzer
// ---------------------------------------------------------------------------

/// Computes pacing statistics from a list of per-shot frame durations.
#[derive(Debug, Clone)]
pub struct PacingAnalyzer {
    /// Frame rate used to convert frames to seconds.
    fps: f64,
}

impl Default for PacingAnalyzer {
    fn default() -> Self {
        Self { fps: 24.0 }
    }
}

impl PacingAnalyzer {
    /// Creates a new analyzer with the specified frame rate.
    #[must_use]
    pub fn new(fps: f64) -> Self {
        Self {
            fps: if fps > 0.0 { fps } else { 24.0 },
        }
    }

    /// Returns the configured frame rate.
    #[must_use]
    pub fn fps(&self) -> f64 {
        self.fps
    }

    /// Computes the average shot length in frames.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn compute_asl(durations: &[u64]) -> f64 {
        if durations.is_empty() {
            return 0.0;
        }
        let sum: u64 = durations.iter().sum();
        sum as f64 / durations.len() as f64
    }

    /// Computes the median of a slice of frame durations.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn compute_median(durations: &[u64]) -> f64 {
        if durations.is_empty() {
            return 0.0;
        }
        let mut sorted: Vec<u64> = durations.to_vec();
        sorted.sort_unstable();
        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            (sorted[mid - 1] as f64 + sorted[mid] as f64) / 2.0
        } else {
            sorted[mid] as f64
        }
    }

    /// Computes population standard deviation.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn compute_std_dev(durations: &[u64], mean: f64) -> f64 {
        if durations.is_empty() {
            return 0.0;
        }
        let variance: f64 = durations
            .iter()
            .map(|&d| {
                let diff = d as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / durations.len() as f64;
        variance.sqrt()
    }

    /// Computes cuts per minute given shot count and total duration in seconds.
    #[must_use]
    pub fn cuts_per_minute(shot_count: usize, total_seconds: f64) -> f64 {
        if total_seconds <= 0.0 || shot_count <= 1 {
            return 0.0;
        }
        // Number of cuts = shot_count - 1
        #[allow(clippy::cast_precision_loss)]
        let cuts = (shot_count - 1) as f64;
        cuts / total_seconds * 60.0
    }

    /// Produces a full [`PacingProfile`] from a list of per-shot frame durations.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn analyze(&self, durations: &[u64]) -> PacingProfile {
        if durations.is_empty() {
            return PacingProfile::default();
        }

        let asl_frames = Self::compute_asl(durations);
        let asl_seconds = asl_frames / self.fps;
        let median = Self::compute_median(durations);
        let std_dev = Self::compute_std_dev(durations, asl_frames);
        let min_frames = *durations.iter().min().unwrap_or(&0);
        let max_frames = *durations.iter().max().unwrap_or(&0);
        let total_frames: u64 = durations.iter().sum();
        let total_seconds = total_frames as f64 / self.fps;
        let cpm = Self::cuts_per_minute(durations.len(), total_seconds);

        PacingProfile {
            shot_count: durations.len(),
            asl_frames,
            asl_seconds,
            median_frames: median,
            std_dev_frames: std_dev,
            min_frames,
            max_frames,
            level: PacingLevel::from_asl_seconds(asl_seconds),
            cuts_per_minute: cpm,
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- PacingLevel --------------------------------------------------------

    #[test]
    fn test_pacing_level_label() {
        assert_eq!(PacingLevel::Rapid.label(), "Rapid");
        assert_eq!(PacingLevel::Lingering.label(), "Lingering");
    }

    #[test]
    fn test_pacing_level_display() {
        assert_eq!(format!("{}", PacingLevel::Fast), "Fast");
    }

    #[test]
    fn test_pacing_level_from_asl() {
        assert_eq!(PacingLevel::from_asl_seconds(1.0), PacingLevel::Rapid);
        assert_eq!(PacingLevel::from_asl_seconds(3.0), PacingLevel::Fast);
        assert_eq!(PacingLevel::from_asl_seconds(6.0), PacingLevel::Moderate);
        assert_eq!(PacingLevel::from_asl_seconds(10.0), PacingLevel::Slow);
        assert_eq!(PacingLevel::from_asl_seconds(20.0), PacingLevel::Lingering);
    }

    #[test]
    fn test_pacing_level_all() {
        assert_eq!(PacingLevel::all().len(), 5);
    }

    // -- PacingAnalyzer: compute_asl ----------------------------------------

    #[test]
    fn test_asl_empty() {
        assert_eq!(PacingAnalyzer::compute_asl(&[]), 0.0);
    }

    #[test]
    fn test_asl_single() {
        assert!((PacingAnalyzer::compute_asl(&[48]) - 48.0).abs() < 1e-6);
    }

    #[test]
    fn test_asl_multiple() {
        let asl = PacingAnalyzer::compute_asl(&[24, 48, 72]);
        assert!((asl - 48.0).abs() < 1e-6);
    }

    // -- PacingAnalyzer: compute_median -------------------------------------

    #[test]
    fn test_median_empty() {
        assert_eq!(PacingAnalyzer::compute_median(&[]), 0.0);
    }

    #[test]
    fn test_median_odd() {
        assert!((PacingAnalyzer::compute_median(&[10, 30, 20]) - 20.0).abs() < 1e-6);
    }

    #[test]
    fn test_median_even() {
        let m = PacingAnalyzer::compute_median(&[10, 20, 30, 40]);
        assert!((m - 25.0).abs() < 1e-6);
    }

    // -- PacingAnalyzer: compute_std_dev ------------------------------------

    #[test]
    fn test_std_dev_empty() {
        assert_eq!(PacingAnalyzer::compute_std_dev(&[], 0.0), 0.0);
    }

    #[test]
    fn test_std_dev_uniform() {
        // All same => std_dev = 0
        assert!(PacingAnalyzer::compute_std_dev(&[48, 48, 48], 48.0).abs() < 1e-6);
    }

    // -- PacingAnalyzer: cuts_per_minute ------------------------------------

    #[test]
    fn test_cpm_zero_shots() {
        assert_eq!(PacingAnalyzer::cuts_per_minute(0, 60.0), 0.0);
    }

    #[test]
    fn test_cpm_single_shot() {
        assert_eq!(PacingAnalyzer::cuts_per_minute(1, 60.0), 0.0);
    }

    #[test]
    fn test_cpm_normal() {
        // 31 shots in 60 seconds = 30 cuts per minute
        let cpm = PacingAnalyzer::cuts_per_minute(31, 60.0);
        assert!((cpm - 30.0).abs() < 1e-6);
    }

    // -- PacingAnalyzer: analyze --------------------------------------------

    #[test]
    fn test_analyze_empty() {
        let a = PacingAnalyzer::new(24.0);
        let p = a.analyze(&[]);
        assert_eq!(p.shot_count, 0);
        assert_eq!(p.level, PacingLevel::Moderate);
    }

    #[test]
    fn test_analyze_rapid() {
        let a = PacingAnalyzer::new(24.0);
        // 24 frames at 24 fps = 1 second each → Rapid
        let p = a.analyze(&[24, 24, 24, 24, 24]);
        assert_eq!(p.level, PacingLevel::Rapid);
        assert!(p.cuts_per_minute > 0.0);
    }

    #[test]
    fn test_analyze_slow() {
        let a = PacingAnalyzer::new(24.0);
        // 240 frames at 24 fps = 10 seconds each → Slow
        let p = a.analyze(&[240, 240, 240]);
        assert_eq!(p.level, PacingLevel::Slow);
    }
}
