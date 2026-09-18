//! Color continuity checker for assembled video clips.
//!
//! Detects jarring color shifts between consecutive clips in an assembled
//! edit by analyzing:
//!
//! - **Luminance distribution**: Mean and standard deviation of brightness
//! - **Color temperature**: Warm/cool shift detection
//! - **Histogram distance**: Earth Mover / Chi-squared histogram comparison
//! - **Saturation shift**: Changes in overall saturation level
//!
//! Each consecutive clip pair receives a discontinuity score. Pairs exceeding
//! a configurable threshold are flagged with remediation suggestions (e.g.,
//! color grading, transition insertion).
//!
//! # Example
//!
//! ```
//! use oximedia_auto::color_continuity::{ColorContinuityChecker, ContinuityConfig};
//!
//! let config = ContinuityConfig::default();
//! let checker = ColorContinuityChecker::new(config);
//! ```

#![allow(dead_code)]

use crate::error::{AutoError, AutoResult};
use oximedia_core::Timestamp;

// ---------------------------------------------------------------------------
// Color statistics for a single clip / frame group
// ---------------------------------------------------------------------------

/// Aggregate color statistics for a clip or frame group.
#[derive(Debug, Clone)]
pub struct ClipColorStats {
    /// Identifier or index of the clip.
    pub clip_index: usize,
    /// Start timestamp.
    pub start: Timestamp,
    /// End timestamp.
    pub end: Timestamp,
    /// Mean luminance (0.0-1.0).
    pub mean_luminance: f64,
    /// Standard deviation of luminance.
    pub luminance_std: f64,
    /// Mean saturation (0.0-1.0).
    pub mean_saturation: f64,
    /// Estimated color temperature bias (-1.0 cool .. +1.0 warm).
    pub color_temperature: f64,
    /// Luminance histogram (16 bins, each 0.0-1.0 normalized).
    pub luminance_histogram: [f64; 16],
}

impl ClipColorStats {
    /// Create a new stats record with default (neutral) values.
    #[must_use]
    pub fn new(clip_index: usize, start: Timestamp, end: Timestamp) -> Self {
        Self {
            clip_index,
            start,
            end,
            mean_luminance: 0.5,
            luminance_std: 0.1,
            mean_saturation: 0.5,
            color_temperature: 0.0,
            luminance_histogram: [1.0 / 16.0; 16],
        }
    }

    /// Build color stats from a grayscale frame buffer.
    ///
    /// This is a simplified analysis that works on luma-only data.
    pub fn from_luma_frame(
        clip_index: usize,
        start: Timestamp,
        end: Timestamp,
        pixels: &[u8],
    ) -> Self {
        let n = pixels.len();
        if n == 0 {
            return Self::new(clip_index, start, end);
        }

        // Mean luminance
        let sum: u64 = pixels.iter().map(|&p| p as u64).sum();
        let mean = sum as f64 / (n as f64 * 255.0);

        // Std dev of luminance
        let mean_raw = sum as f64 / n as f64;
        let var: f64 = pixels
            .iter()
            .map(|&p| {
                let d = p as f64 - mean_raw;
                d * d
            })
            .sum::<f64>()
            / n as f64;
        let std = (var.sqrt() / 255.0).min(1.0);

        // Histogram (16 bins)
        let mut histogram = [0u64; 16];
        for &p in pixels {
            let bin = (p as usize * 15 / 255).min(15);
            histogram[bin] += 1;
        }
        let mut hist_norm = [0.0f64; 16];
        for (i, &count) in histogram.iter().enumerate() {
            hist_norm[i] = count as f64 / n as f64;
        }

        Self {
            clip_index,
            start,
            end,
            mean_luminance: mean,
            luminance_std: std,
            mean_saturation: 0.5, // would need chroma for real value
            color_temperature: 0.0,
            luminance_histogram: hist_norm,
        }
    }
}

// ---------------------------------------------------------------------------
// Discontinuity report
// ---------------------------------------------------------------------------

/// Severity level for a detected discontinuity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    /// Barely noticeable; no action needed.
    Minor,
    /// Noticeable; consider a transition.
    Moderate,
    /// Jarring; color grading recommended.
    Major,
    /// Extremely jarring; strongly recommend fix.
    Critical,
}

impl Severity {
    /// Create from a raw discontinuity score (0.0-1.0+).
    #[must_use]
    pub fn from_score(score: f64) -> Self {
        if score < 0.15 {
            Self::Minor
        } else if score < 0.35 {
            Self::Moderate
        } else if score < 0.60 {
            Self::Major
        } else {
            Self::Critical
        }
    }
}

/// Suggested remediation for a detected color discontinuity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Remediation {
    /// No action needed.
    None,
    /// Insert a dissolve or fade transition to mask the shift.
    InsertTransition,
    /// Apply color grading to one or both clips.
    ColorGrade,
    /// Add a dip-to-black / dip-to-white between clips.
    DipToBlack,
    /// Re-order clips to reduce the visual jump.
    Reorder,
}

/// A single discontinuity detected between two consecutive clips.
#[derive(Debug, Clone)]
pub struct Discontinuity {
    /// Index of the first clip in the pair.
    pub clip_a_index: usize,
    /// Index of the second clip in the pair.
    pub clip_b_index: usize,
    /// Overall discontinuity score (0.0 = perfect match, 1.0+ = extreme).
    pub score: f64,
    /// Severity classification.
    pub severity: Severity,
    /// Individual metric contributions.
    pub luminance_delta: f64,
    /// Saturation delta.
    pub saturation_delta: f64,
    /// Color temperature delta.
    pub temperature_delta: f64,
    /// Histogram distance.
    pub histogram_distance: f64,
    /// Suggested remediation.
    pub remediation: Remediation,
    /// Human-readable description.
    pub description: String,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the color continuity checker.
#[derive(Debug, Clone)]
pub struct ContinuityConfig {
    /// Threshold above which a pair is flagged (0.0-1.0).
    pub flag_threshold: f64,
    /// Weight for luminance difference.
    pub luminance_weight: f64,
    /// Weight for saturation difference.
    pub saturation_weight: f64,
    /// Weight for color temperature difference.
    pub temperature_weight: f64,
    /// Weight for histogram distance.
    pub histogram_weight: f64,
}

impl Default for ContinuityConfig {
    fn default() -> Self {
        Self {
            flag_threshold: 0.20,
            luminance_weight: 1.0,
            saturation_weight: 0.8,
            temperature_weight: 0.9,
            histogram_weight: 1.2,
        }
    }
}

impl ContinuityConfig {
    /// Create default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the flag threshold.
    #[must_use]
    pub fn with_threshold(mut self, t: f64) -> Self {
        self.flag_threshold = t.clamp(0.0, 1.0);
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> AutoResult<()> {
        if !(0.0..=1.0).contains(&self.flag_threshold) {
            return Err(AutoError::InvalidThreshold {
                threshold: self.flag_threshold,
                min: 0.0,
                max: 1.0,
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Distance metrics
// ---------------------------------------------------------------------------

/// Chi-squared histogram distance in [0, 1].
fn chi_squared_distance(a: &[f64; 16], b: &[f64; 16]) -> f64 {
    let mut dist = 0.0;
    for i in 0..16 {
        let sum = a[i] + b[i];
        if sum > 1e-10 {
            let diff = a[i] - b[i];
            dist += diff * diff / sum;
        }
    }
    // chi-squared ranges 0..2; normalize to 0..1
    (dist / 2.0).min(1.0)
}

/// Suggest remediation based on the dominant cause of discontinuity.
fn suggest_remediation(disc: &Discontinuity, threshold: f64) -> Remediation {
    if disc.score < threshold {
        return Remediation::None;
    }

    // Critical discontinuities
    if disc.severity == Severity::Critical {
        return Remediation::DipToBlack;
    }

    // If histogram is the main problem, color grading helps
    if disc.histogram_distance > disc.luminance_delta
        && disc.histogram_distance > disc.temperature_delta
    {
        return Remediation::ColorGrade;
    }

    // Temperature shift → color grade
    if disc.temperature_delta > 0.3 {
        return Remediation::ColorGrade;
    }

    // Moderate luminance shift → dissolve
    Remediation::InsertTransition
}

/// Build a human-readable description of the discontinuity.
fn build_description(disc: &Discontinuity) -> String {
    let mut parts = Vec::new();

    if disc.luminance_delta > 0.10 {
        parts.push(format!(
            "brightness shift {:.0}%",
            disc.luminance_delta * 100.0
        ));
    }
    if disc.saturation_delta > 0.10 {
        parts.push(format!(
            "saturation shift {:.0}%",
            disc.saturation_delta * 100.0
        ));
    }
    if disc.temperature_delta > 0.15 {
        parts.push(format!(
            "color temp shift {:.0}%",
            disc.temperature_delta * 100.0
        ));
    }
    if disc.histogram_distance > 0.10 {
        parts.push(format!(
            "histogram dist {:.0}%",
            disc.histogram_distance * 100.0
        ));
    }

    if parts.is_empty() {
        format!(
            "Clip {} -> {}: minor shift (score {:.2})",
            disc.clip_a_index, disc.clip_b_index, disc.score
        )
    } else {
        format!(
            "Clip {} -> {}: {}",
            disc.clip_a_index,
            disc.clip_b_index,
            parts.join(", ")
        )
    }
}

// ---------------------------------------------------------------------------
// ColorContinuityChecker
// ---------------------------------------------------------------------------

/// Checks color continuity across a sequence of clips.
pub struct ColorContinuityChecker {
    config: ContinuityConfig,
}

impl ColorContinuityChecker {
    /// Create a new checker with the given configuration.
    #[must_use]
    pub fn new(config: ContinuityConfig) -> Self {
        Self { config }
    }

    /// Analyze a sequence of clip color stats and return all detected
    /// discontinuities (including those below the flag threshold, with
    /// `Severity::Minor`).
    ///
    /// # Errors
    ///
    /// Returns an error if configuration is invalid or fewer than 2 clips
    /// are provided.
    pub fn check(&self, clips: &[ClipColorStats]) -> AutoResult<Vec<Discontinuity>> {
        self.config.validate()?;

        if clips.len() < 2 {
            return Err(AutoError::insufficient_data(
                "Need at least 2 clips for continuity check",
            ));
        }

        let total_weight = self.config.luminance_weight
            + self.config.saturation_weight
            + self.config.temperature_weight
            + self.config.histogram_weight;

        let mut results = Vec::with_capacity(clips.len() - 1);

        for pair in clips.windows(2) {
            let a = &pair[0];
            let b = &pair[1];

            let lum_delta = (a.mean_luminance - b.mean_luminance).abs();
            let sat_delta = (a.mean_saturation - b.mean_saturation).abs();
            let temp_delta = (a.color_temperature - b.color_temperature).abs();
            let hist_dist = chi_squared_distance(&a.luminance_histogram, &b.luminance_histogram);

            let weighted_score = if total_weight > 0.0 {
                (lum_delta * self.config.luminance_weight
                    + sat_delta * self.config.saturation_weight
                    + temp_delta * self.config.temperature_weight
                    + hist_dist * self.config.histogram_weight)
                    / total_weight
            } else {
                0.0
            };

            let severity = Severity::from_score(weighted_score);

            let mut disc = Discontinuity {
                clip_a_index: a.clip_index,
                clip_b_index: b.clip_index,
                score: weighted_score,
                severity,
                luminance_delta: lum_delta,
                saturation_delta: sat_delta,
                temperature_delta: temp_delta,
                histogram_distance: hist_dist,
                remediation: Remediation::None,
                description: String::new(),
            };

            disc.remediation = suggest_remediation(&disc, self.config.flag_threshold);
            disc.description = build_description(&disc);

            results.push(disc);
        }

        Ok(results)
    }

    /// Return only flagged discontinuities (above threshold).
    ///
    /// # Errors
    ///
    /// Returns an error if `check` fails.
    pub fn flagged(&self, clips: &[ClipColorStats]) -> AutoResult<Vec<Discontinuity>> {
        let all = self.check(clips)?;
        Ok(all
            .into_iter()
            .filter(|d| d.score >= self.config.flag_threshold)
            .collect())
    }

    /// Compute the overall continuity score for the entire sequence (0.0 =
    /// perfect continuity, 1.0 = very discontinuous).
    ///
    /// # Errors
    ///
    /// Returns an error if `check` fails.
    pub fn overall_score(&self, clips: &[ClipColorStats]) -> AutoResult<f64> {
        let all = self.check(clips)?;
        if all.is_empty() {
            return Ok(0.0);
        }
        let sum: f64 = all.iter().map(|d| d.score).sum();
        Ok(sum / all.len() as f64)
    }

    /// Get the current configuration.
    #[must_use]
    pub const fn config(&self) -> &ContinuityConfig {
        &self.config
    }
}

impl Default for ColorContinuityChecker {
    fn default() -> Self {
        Self::new(ContinuityConfig::default())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_core::Rational;

    fn ts(ms: i64) -> Timestamp {
        Timestamp::new(ms, Rational::new(1, 1000))
    }

    fn make_stats(
        index: usize,
        start_ms: i64,
        end_ms: i64,
        luminance: f64,
        saturation: f64,
        temperature: f64,
    ) -> ClipColorStats {
        ClipColorStats {
            clip_index: index,
            start: ts(start_ms),
            end: ts(end_ms),
            mean_luminance: luminance,
            luminance_std: 0.1,
            mean_saturation: saturation,
            color_temperature: temperature,
            luminance_histogram: [1.0 / 16.0; 16],
        }
    }

    #[test]
    fn test_config_default_valid() {
        let cfg = ContinuityConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_invalid_threshold() {
        let cfg = ContinuityConfig::default().with_threshold(1.5);
        // clamp makes it 1.0 which is valid
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_check_needs_two_clips() {
        let checker = ColorContinuityChecker::default();
        let result = checker.check(&[make_stats(0, 0, 1000, 0.5, 0.5, 0.0)]);
        assert!(result.is_err());
    }

    #[test]
    fn test_check_identical_clips_low_score() {
        let clips = vec![
            make_stats(0, 0, 5000, 0.5, 0.5, 0.0),
            make_stats(1, 5000, 10000, 0.5, 0.5, 0.0),
        ];
        let checker = ColorContinuityChecker::default();
        let results = checker.check(&clips).expect("should succeed");
        assert_eq!(results.len(), 1);
        assert!(
            results[0].score < 0.01,
            "identical clips should have near-zero score: {}",
            results[0].score
        );
        assert_eq!(results[0].severity, Severity::Minor);
    }

    #[test]
    fn test_check_luminance_shift_detected() {
        let clips = vec![
            make_stats(0, 0, 5000, 0.2, 0.5, 0.0),
            make_stats(1, 5000, 10000, 0.8, 0.5, 0.0),
        ];
        let checker = ColorContinuityChecker::default();
        let results = checker.check(&clips).expect("should succeed");
        assert!(results[0].luminance_delta > 0.5);
        assert!(results[0].score > 0.1);
    }

    #[test]
    fn test_check_temperature_shift_detected() {
        let clips = vec![
            make_stats(0, 0, 5000, 0.5, 0.5, -0.8),
            make_stats(1, 5000, 10000, 0.5, 0.5, 0.8),
        ];
        let checker = ColorContinuityChecker::default();
        let results = checker.check(&clips).expect("should succeed");
        assert!(results[0].temperature_delta > 1.0);
    }

    #[test]
    fn test_check_saturation_shift_detected() {
        let clips = vec![
            make_stats(0, 0, 5000, 0.5, 0.1, 0.0),
            make_stats(1, 5000, 10000, 0.5, 0.9, 0.0),
        ];
        let checker = ColorContinuityChecker::default();
        let results = checker.check(&clips).expect("should succeed");
        assert!(results[0].saturation_delta > 0.5);
    }

    #[test]
    fn test_flagged_filters_low_scores() {
        let clips = vec![
            make_stats(0, 0, 5000, 0.5, 0.5, 0.0),
            make_stats(1, 5000, 10000, 0.51, 0.5, 0.0), // tiny change
            make_stats(2, 10000, 15000, 0.9, 0.1, 0.8), // big change
        ];
        let checker = ColorContinuityChecker::default();
        let flagged = checker.flagged(&clips).expect("should succeed");
        // Only the second pair should be flagged
        assert_eq!(flagged.len(), 1);
        assert_eq!(flagged[0].clip_a_index, 1);
        assert_eq!(flagged[0].clip_b_index, 2);
    }

    #[test]
    fn test_overall_score_perfect() {
        let clips = vec![
            make_stats(0, 0, 5000, 0.5, 0.5, 0.0),
            make_stats(1, 5000, 10000, 0.5, 0.5, 0.0),
        ];
        let checker = ColorContinuityChecker::default();
        let score = checker.overall_score(&clips).expect("should succeed");
        assert!(score < 0.01);
    }

    #[test]
    fn test_overall_score_discontinuous() {
        let clips = vec![
            make_stats(0, 0, 5000, 0.1, 0.1, -0.5),
            make_stats(1, 5000, 10000, 0.9, 0.9, 0.5),
        ];
        let checker = ColorContinuityChecker::default();
        let score = checker.overall_score(&clips).expect("should succeed");
        assert!(score > 0.2);
    }

    #[test]
    fn test_severity_from_score() {
        assert_eq!(Severity::from_score(0.05), Severity::Minor);
        assert_eq!(Severity::from_score(0.20), Severity::Moderate);
        assert_eq!(Severity::from_score(0.40), Severity::Major);
        assert_eq!(Severity::from_score(0.70), Severity::Critical);
    }

    #[test]
    fn test_remediation_none_below_threshold() {
        let disc = Discontinuity {
            clip_a_index: 0,
            clip_b_index: 1,
            score: 0.05,
            severity: Severity::Minor,
            luminance_delta: 0.03,
            saturation_delta: 0.02,
            temperature_delta: 0.01,
            histogram_distance: 0.01,
            remediation: Remediation::None,
            description: String::new(),
        };
        let rem = suggest_remediation(&disc, 0.20);
        assert_eq!(rem, Remediation::None);
    }

    #[test]
    fn test_remediation_critical_dip_to_black() {
        let disc = Discontinuity {
            clip_a_index: 0,
            clip_b_index: 1,
            score: 0.75,
            severity: Severity::Critical,
            luminance_delta: 0.5,
            saturation_delta: 0.5,
            temperature_delta: 0.5,
            histogram_distance: 0.5,
            remediation: Remediation::None,
            description: String::new(),
        };
        let rem = suggest_remediation(&disc, 0.20);
        assert_eq!(rem, Remediation::DipToBlack);
    }

    #[test]
    fn test_remediation_histogram_color_grade() {
        let disc = Discontinuity {
            clip_a_index: 0,
            clip_b_index: 1,
            score: 0.30,
            severity: Severity::Moderate,
            luminance_delta: 0.05,
            saturation_delta: 0.05,
            temperature_delta: 0.05,
            histogram_distance: 0.40,
            remediation: Remediation::None,
            description: String::new(),
        };
        let rem = suggest_remediation(&disc, 0.20);
        assert_eq!(rem, Remediation::ColorGrade);
    }

    #[test]
    fn test_chi_squared_identical() {
        let h = [1.0 / 16.0; 16];
        let d = chi_squared_distance(&h, &h);
        assert!(d < 1e-10);
    }

    #[test]
    fn test_chi_squared_different() {
        let mut a = [0.0; 16];
        a[0] = 1.0; // all mass in bin 0
        let mut b = [0.0; 16];
        b[15] = 1.0; // all mass in bin 15
        let d = chi_squared_distance(&a, &b);
        assert!((d - 1.0).abs() < 1e-6, "maximally different = 1.0: {d}");
    }

    #[test]
    fn test_from_luma_frame_uniform() {
        let pixels = vec![128u8; 1000];
        let stats = ClipColorStats::from_luma_frame(0, ts(0), ts(1000), &pixels);
        assert!((stats.mean_luminance - 128.0 / 255.0).abs() < 0.01);
        assert!(stats.luminance_std < 0.01, "uniform → low std");
    }

    #[test]
    fn test_from_luma_frame_empty() {
        let stats = ClipColorStats::from_luma_frame(0, ts(0), ts(1000), &[]);
        assert!((stats.mean_luminance - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_from_luma_frame_histogram_shape() {
        // All pixels in the dark range
        let pixels = vec![10u8; 500];
        let stats = ClipColorStats::from_luma_frame(0, ts(0), ts(1000), &pixels);
        // Bin 0 should have all the mass
        assert!(
            stats.luminance_histogram[0] > 0.9,
            "dark pixels should cluster in bin 0: {:.3}",
            stats.luminance_histogram[0]
        );
    }

    #[test]
    fn test_build_description_minor() {
        let disc = Discontinuity {
            clip_a_index: 0,
            clip_b_index: 1,
            score: 0.05,
            severity: Severity::Minor,
            luminance_delta: 0.02,
            saturation_delta: 0.01,
            temperature_delta: 0.01,
            histogram_distance: 0.01,
            remediation: Remediation::None,
            description: String::new(),
        };
        let desc = build_description(&disc);
        assert!(desc.contains("minor shift"));
    }

    #[test]
    fn test_build_description_major() {
        let disc = Discontinuity {
            clip_a_index: 2,
            clip_b_index: 3,
            score: 0.5,
            severity: Severity::Major,
            luminance_delta: 0.30,
            saturation_delta: 0.25,
            temperature_delta: 0.20,
            histogram_distance: 0.15,
            remediation: Remediation::ColorGrade,
            description: String::new(),
        };
        let desc = build_description(&disc);
        assert!(desc.contains("brightness shift"));
        assert!(desc.contains("saturation shift"));
        assert!(desc.contains("color temp shift"));
        assert!(desc.contains("histogram dist"));
    }

    #[test]
    fn test_three_clip_sequence() {
        let clips = vec![
            make_stats(0, 0, 5000, 0.4, 0.5, 0.0),
            make_stats(1, 5000, 10000, 0.45, 0.5, 0.0),
            make_stats(2, 10000, 15000, 0.8, 0.5, 0.0),
        ];
        let checker = ColorContinuityChecker::default();
        let results = checker.check(&clips).expect("should succeed");
        assert_eq!(results.len(), 2);
        // First pair: small shift
        assert!(results[0].score < results[1].score);
    }
}
