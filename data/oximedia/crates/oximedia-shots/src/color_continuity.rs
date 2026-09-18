//! Color continuity analysis between consecutive shots.
//!
//! Detects color temperature, grading, and white-balance inconsistencies
//! that break visual continuity. Computes per-channel statistics, correlated
//! color temperature (CCT) estimates, and provides an overall continuity
//! score between shot pairs.

use crate::error::{ShotError, ShotResult};
use crate::frame_buffer::FrameBuffer;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Color statistics for a single frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameColorStats {
    /// Mean red channel value (0-255).
    pub mean_r: f32,
    /// Mean green channel value (0-255).
    pub mean_g: f32,
    /// Mean blue channel value (0-255).
    pub mean_b: f32,
    /// Standard deviation of red channel.
    pub std_r: f32,
    /// Standard deviation of green channel.
    pub std_g: f32,
    /// Standard deviation of blue channel.
    pub std_b: f32,
    /// Estimated correlated color temperature in Kelvin.
    pub color_temperature_k: f32,
    /// Green-magenta tint (positive = green, negative = magenta).
    pub tint: f32,
    /// Overall luminance (BT.601 weighted).
    pub luminance: f32,
    /// Saturation estimate (mean chroma distance from neutral axis).
    pub saturation: f32,
}

/// Result of comparing color continuity between two frames.
#[derive(Debug, Clone)]
pub struct ColorContinuityResult {
    /// Index of shot A.
    pub shot_a: usize,
    /// Index of shot B.
    pub shot_b: usize,
    /// Color stats for shot A.
    pub stats_a: FrameColorStats,
    /// Color stats for shot B.
    pub stats_b: FrameColorStats,
    /// Color temperature difference in Kelvin.
    pub temperature_diff_k: f32,
    /// Tint difference.
    pub tint_diff: f32,
    /// Luminance difference (normalised 0-1).
    pub luminance_diff: f32,
    /// Saturation difference (normalised 0-1).
    pub saturation_diff: f32,
    /// Per-channel mean difference (normalised 0-1).
    pub channel_diff: f32,
    /// Overall continuity score (0.0 = severe mismatch, 1.0 = perfect match).
    pub continuity_score: f32,
    /// Detected issues.
    pub issues: Vec<ColorIssue>,
}

/// A specific color continuity issue.
#[derive(Debug, Clone)]
pub struct ColorIssue {
    /// Issue severity.
    pub severity: ColorIssueSeverity,
    /// Category of the issue.
    pub category: ColorIssueCategory,
    /// Human-readable description.
    pub description: String,
}

/// Severity of a color continuity issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ColorIssueSeverity {
    /// Subtle difference, likely acceptable.
    Info,
    /// Noticeable difference that may need correction.
    Warning,
    /// Severe mismatch that will be visible to viewers.
    Error,
}

/// Category of color continuity issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorIssueCategory {
    /// Color temperature mismatch (warm vs cool).
    Temperature,
    /// White balance tint mismatch (green vs magenta).
    Tint,
    /// Exposure / luminance mismatch.
    Exposure,
    /// Saturation mismatch.
    Saturation,
    /// Overall color grading mismatch.
    Grading,
}

/// Configuration for the color continuity checker.
#[derive(Debug, Clone)]
pub struct ColorContinuityConfig {
    /// Maximum acceptable color temperature difference (Kelvin).
    pub max_temperature_diff_k: f32,
    /// Maximum acceptable tint difference.
    pub max_tint_diff: f32,
    /// Maximum acceptable luminance difference (normalised).
    pub max_luminance_diff: f32,
    /// Maximum acceptable saturation difference (normalised).
    pub max_saturation_diff: f32,
    /// Maximum acceptable per-channel mean difference (normalised).
    pub max_channel_diff: f32,
}

impl Default for ColorContinuityConfig {
    fn default() -> Self {
        Self {
            max_temperature_diff_k: 500.0,
            max_tint_diff: 0.05,
            max_luminance_diff: 0.10,
            max_saturation_diff: 0.10,
            max_channel_diff: 0.08,
        }
    }
}

// ---------------------------------------------------------------------------
// Analyzer
// ---------------------------------------------------------------------------

/// Color continuity analyzer.
pub struct ColorContinuityAnalyzer {
    config: ColorContinuityConfig,
}

impl Default for ColorContinuityAnalyzer {
    fn default() -> Self {
        Self::new(ColorContinuityConfig::default())
    }
}

impl ColorContinuityAnalyzer {
    /// Create a new analyzer with the given configuration.
    #[must_use]
    pub fn new(config: ColorContinuityConfig) -> Self {
        Self { config }
    }

    /// Compute color statistics for a frame.
    ///
    /// # Errors
    ///
    /// Returns error if frame has fewer than 3 channels.
    pub fn compute_stats(&self, frame: &FrameBuffer) -> ShotResult<FrameColorStats> {
        let (h, w, ch) = frame.dim();
        if ch < 3 {
            return Err(ShotError::InvalidFrame(
                "Frame must have at least 3 channels".to_string(),
            ));
        }
        let n = (h * w) as f64;
        if n < 1.0 {
            return Err(ShotError::InvalidFrame("Frame is empty".to_string()));
        }

        let mut sum_r = 0.0_f64;
        let mut sum_g = 0.0_f64;
        let mut sum_b = 0.0_f64;
        let mut sum_r2 = 0.0_f64;
        let mut sum_g2 = 0.0_f64;
        let mut sum_b2 = 0.0_f64;
        let mut sum_sat = 0.0_f64;

        for y in 0..h {
            for x in 0..w {
                let r = f64::from(frame.get(y, x, 0));
                let g = f64::from(frame.get(y, x, 1));
                let b = f64::from(frame.get(y, x, 2));

                sum_r += r;
                sum_g += g;
                sum_b += b;
                sum_r2 += r * r;
                sum_g2 += g * g;
                sum_b2 += b * b;

                // Saturation: distance from neutral axis in RGB cube
                let mean_rgb = (r + g + b) / 3.0;
                let dr = r - mean_rgb;
                let dg = g - mean_rgb;
                let db = b - mean_rgb;
                sum_sat += (dr * dr + dg * dg + db * db).sqrt();
            }
        }

        let mean_r = (sum_r / n) as f32;
        let mean_g = (sum_g / n) as f32;
        let mean_b = (sum_b / n) as f32;
        let std_r = ((sum_r2 / n - (sum_r / n).powi(2)).max(0.0).sqrt()) as f32;
        let std_g = ((sum_g2 / n - (sum_g / n).powi(2)).max(0.0).sqrt()) as f32;
        let std_b = ((sum_b2 / n - (sum_b / n).powi(2)).max(0.0).sqrt()) as f32;
        let luminance =
            (0.299 * mean_r as f64 + 0.587 * mean_g as f64 + 0.114 * mean_b as f64) as f32 / 255.0;
        let saturation = ((sum_sat / n) / 255.0) as f32;

        // Estimate CCT using McCamy's approximation from CIE xy chromaticity
        let color_temperature_k = estimate_cct(mean_r, mean_g, mean_b);
        let tint = estimate_tint(mean_r, mean_g, mean_b);

        Ok(FrameColorStats {
            mean_r,
            mean_g,
            mean_b,
            std_r,
            std_g,
            std_b,
            color_temperature_k,
            tint,
            luminance,
            saturation,
        })
    }

    /// Check color continuity between two frames (representative of two shots).
    ///
    /// # Errors
    ///
    /// Returns error if frames are invalid.
    pub fn check_continuity(
        &self,
        frame_a: &FrameBuffer,
        frame_b: &FrameBuffer,
        shot_a: usize,
        shot_b: usize,
    ) -> ShotResult<ColorContinuityResult> {
        let stats_a = self.compute_stats(frame_a)?;
        let stats_b = self.compute_stats(frame_b)?;

        let temperature_diff_k = (stats_a.color_temperature_k - stats_b.color_temperature_k).abs();
        let tint_diff = (stats_a.tint - stats_b.tint).abs();
        let luminance_diff = (stats_a.luminance - stats_b.luminance).abs();
        let saturation_diff = (stats_a.saturation - stats_b.saturation).abs();

        let channel_diff = {
            let dr = (stats_a.mean_r - stats_b.mean_r).abs() / 255.0;
            let dg = (stats_a.mean_g - stats_b.mean_g).abs() / 255.0;
            let db = (stats_a.mean_b - stats_b.mean_b).abs() / 255.0;
            (dr + dg + db) / 3.0
        };

        let mut issues = Vec::new();

        // Temperature check
        if temperature_diff_k > self.config.max_temperature_diff_k * 2.0 {
            issues.push(ColorIssue {
                severity: ColorIssueSeverity::Error,
                category: ColorIssueCategory::Temperature,
                description: format!(
                    "Severe color temperature mismatch: {:.0}K difference ({:.0}K vs {:.0}K)",
                    temperature_diff_k, stats_a.color_temperature_k, stats_b.color_temperature_k
                ),
            });
        } else if temperature_diff_k > self.config.max_temperature_diff_k {
            issues.push(ColorIssue {
                severity: ColorIssueSeverity::Warning,
                category: ColorIssueCategory::Temperature,
                description: format!(
                    "Color temperature difference: {:.0}K ({:.0}K vs {:.0}K)",
                    temperature_diff_k, stats_a.color_temperature_k, stats_b.color_temperature_k
                ),
            });
        }

        // Tint check
        if tint_diff > self.config.max_tint_diff * 2.0 {
            issues.push(ColorIssue {
                severity: ColorIssueSeverity::Error,
                category: ColorIssueCategory::Tint,
                description: format!("Severe tint mismatch: {tint_diff:.3}"),
            });
        } else if tint_diff > self.config.max_tint_diff {
            issues.push(ColorIssue {
                severity: ColorIssueSeverity::Warning,
                category: ColorIssueCategory::Tint,
                description: format!("Tint difference: {tint_diff:.3}"),
            });
        }

        // Exposure check
        if luminance_diff > self.config.max_luminance_diff * 2.0 {
            issues.push(ColorIssue {
                severity: ColorIssueSeverity::Error,
                category: ColorIssueCategory::Exposure,
                description: format!("Severe exposure mismatch: {luminance_diff:.3}"),
            });
        } else if luminance_diff > self.config.max_luminance_diff {
            issues.push(ColorIssue {
                severity: ColorIssueSeverity::Warning,
                category: ColorIssueCategory::Exposure,
                description: format!("Exposure difference: {luminance_diff:.3}"),
            });
        }

        // Saturation check
        if saturation_diff > self.config.max_saturation_diff * 2.0 {
            issues.push(ColorIssue {
                severity: ColorIssueSeverity::Error,
                category: ColorIssueCategory::Saturation,
                description: format!("Severe saturation mismatch: {saturation_diff:.3}"),
            });
        } else if saturation_diff > self.config.max_saturation_diff {
            issues.push(ColorIssue {
                severity: ColorIssueSeverity::Warning,
                category: ColorIssueCategory::Saturation,
                description: format!("Saturation difference: {saturation_diff:.3}"),
            });
        }

        // Overall grading check (channel balance)
        if channel_diff > self.config.max_channel_diff * 2.0 {
            issues.push(ColorIssue {
                severity: ColorIssueSeverity::Error,
                category: ColorIssueCategory::Grading,
                description: format!("Severe color grading mismatch: {channel_diff:.3}"),
            });
        } else if channel_diff > self.config.max_channel_diff {
            issues.push(ColorIssue {
                severity: ColorIssueSeverity::Warning,
                category: ColorIssueCategory::Grading,
                description: format!("Color grading difference: {channel_diff:.3}"),
            });
        }

        // Combined continuity score
        let temp_score = (1.0
            - (temperature_diff_k / (self.config.max_temperature_diff_k * 3.0)).min(1.0))
        .max(0.0);
        let tint_score = (1.0 - (tint_diff / (self.config.max_tint_diff * 3.0)).min(1.0)).max(0.0);
        let lum_score =
            (1.0 - (luminance_diff / (self.config.max_luminance_diff * 3.0)).min(1.0)).max(0.0);
        let sat_score =
            (1.0 - (saturation_diff / (self.config.max_saturation_diff * 3.0)).min(1.0)).max(0.0);
        let chan_score =
            (1.0 - (channel_diff / (self.config.max_channel_diff * 3.0)).min(1.0)).max(0.0);

        let continuity_score = (temp_score * 0.25
            + tint_score * 0.15
            + lum_score * 0.25
            + sat_score * 0.15
            + chan_score * 0.20)
            .clamp(0.0, 1.0);

        Ok(ColorContinuityResult {
            shot_a,
            shot_b,
            stats_a,
            stats_b,
            temperature_diff_k,
            tint_diff,
            luminance_diff,
            saturation_diff,
            channel_diff,
            continuity_score,
            issues,
        })
    }

    /// Check color continuity across a sequence of shots.
    ///
    /// Each frame in `frames` is the representative frame for a shot.
    ///
    /// # Errors
    ///
    /// Returns error if any frame is invalid.
    pub fn check_sequence(&self, frames: &[FrameBuffer]) -> ShotResult<Vec<ColorContinuityResult>> {
        let mut results = Vec::new();
        for i in 1..frames.len() {
            let result = self.check_continuity(&frames[i - 1], &frames[i], i - 1, i)?;
            results.push(result);
        }
        Ok(results)
    }

    /// Get a reference to the configuration.
    #[must_use]
    pub fn config(&self) -> &ColorContinuityConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// CCT / Tint estimation helpers
// ---------------------------------------------------------------------------

/// Estimate correlated color temperature from mean RGB values.
///
/// Uses a simplified sRGB-to-CIE-xy conversion followed by McCamy's
/// approximation. This is approximate but adequate for detecting
/// relative shifts between shots.
fn estimate_cct(mean_r: f32, mean_g: f32, mean_b: f32) -> f32 {
    // Linearise sRGB (approximate gamma 2.2)
    let r_lin = (mean_r as f64 / 255.0).powf(2.2);
    let g_lin = (mean_g as f64 / 255.0).powf(2.2);
    let b_lin = (mean_b as f64 / 255.0).powf(2.2);

    // sRGB to CIE XYZ (D65 matrix)
    let x = 0.4124 * r_lin + 0.3576 * g_lin + 0.1805 * b_lin;
    let y = 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin;
    let z = 0.0193 * r_lin + 0.1192 * g_lin + 0.9505 * b_lin;

    let sum = x + y + z;
    if sum < 1e-10 {
        return 6500.0; // Default daylight for black
    }

    let cx = x / sum;
    let cy = y / sum;

    // McCamy's approximation
    let n = (cx - 0.3320) / (0.1858 - cy);
    let cct = 449.0 * n * n * n + 3525.0 * n * n + 6823.3 * n + 5520.33;

    // Clamp to reasonable range
    cct.clamp(1000.0, 25000.0) as f32
}

/// Estimate green-magenta tint from mean RGB.
///
/// Positive = green cast, negative = magenta cast. Normalised to roughly
/// [-1, 1] range.
fn estimate_tint(mean_r: f32, mean_g: f32, mean_b: f32) -> f32 {
    let avg_rb = (mean_r + mean_b) / 2.0;
    if avg_rb < f32::EPSILON {
        return 0.0;
    }
    // Green excess relative to red-blue average
    ((mean_g - avg_rb) / avg_rb).clamp(-1.0, 1.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_color_frame(r: u8, g: u8, b: u8, h: usize, w: usize) -> FrameBuffer {
        let mut frame = FrameBuffer::zeros(h, w, 3);
        for y in 0..h {
            for x in 0..w {
                frame.set(y, x, 0, r);
                frame.set(y, x, 1, g);
                frame.set(y, x, 2, b);
            }
        }
        frame
    }

    #[test]
    fn test_compute_stats_uniform_gray() {
        let analyzer = ColorContinuityAnalyzer::default();
        let frame = FrameBuffer::from_elem(50, 50, 3, 128);
        let stats = analyzer
            .compute_stats(&frame)
            .expect("should succeed in test");
        assert!((stats.mean_r - 128.0).abs() < 0.01);
        assert!((stats.mean_g - 128.0).abs() < 0.01);
        assert!((stats.mean_b - 128.0).abs() < 0.01);
        assert!(stats.std_r < 0.01);
        assert!(stats.std_g < 0.01);
        assert!(stats.std_b < 0.01);
        assert!(
            stats.saturation < 0.01,
            "gray frame should have ~zero saturation"
        );
    }

    #[test]
    fn test_compute_stats_warm_frame() {
        let analyzer = ColorContinuityAnalyzer::default();
        let frame = make_color_frame(220, 180, 120, 50, 50);
        let stats = analyzer
            .compute_stats(&frame)
            .expect("should succeed in test");
        assert!(stats.mean_r > stats.mean_b, "warm frame: R > B");
        // CCT should be lower (warmer) than daylight
        assert!(stats.color_temperature_k < 6500.0);
    }

    #[test]
    fn test_compute_stats_cool_frame() {
        let analyzer = ColorContinuityAnalyzer::default();
        let frame = make_color_frame(120, 150, 220, 50, 50);
        let stats = analyzer
            .compute_stats(&frame)
            .expect("should succeed in test");
        assert!(stats.mean_b > stats.mean_r, "cool frame: B > R");
        assert!(stats.color_temperature_k > 6500.0);
    }

    #[test]
    fn test_compute_stats_invalid_frame() {
        let analyzer = ColorContinuityAnalyzer::default();
        let frame = FrameBuffer::zeros(50, 50, 1);
        assert!(analyzer.compute_stats(&frame).is_err());
    }

    #[test]
    fn test_compute_stats_empty_frame() {
        let analyzer = ColorContinuityAnalyzer::default();
        let frame = FrameBuffer::zeros(0, 0, 3);
        assert!(analyzer.compute_stats(&frame).is_err());
    }

    #[test]
    fn test_check_continuity_identical() {
        let analyzer = ColorContinuityAnalyzer::default();
        let frame = FrameBuffer::from_elem(50, 50, 3, 128);
        let result = analyzer
            .check_continuity(&frame, &frame, 0, 1)
            .expect("should succeed in test");
        assert!(
            result.continuity_score > 0.95,
            "identical frames should score ~1.0"
        );
        assert!(
            result.issues.is_empty(),
            "identical frames should have no issues"
        );
        assert!(result.temperature_diff_k < 1.0);
        assert!(result.tint_diff < 0.001);
        assert!(result.luminance_diff < 0.001);
    }

    #[test]
    fn test_check_continuity_warm_vs_cool() {
        let analyzer = ColorContinuityAnalyzer::default();
        let warm = make_color_frame(255, 180, 100, 50, 50);
        let cool = make_color_frame(100, 150, 255, 50, 50);
        let result = analyzer
            .check_continuity(&warm, &cool, 0, 1)
            .expect("should succeed in test");
        assert!(
            result.temperature_diff_k > 100.0,
            "warm vs cool should have high temp diff"
        );
        assert!(!result.issues.is_empty(), "warm vs cool should have issues");
        assert!(result.continuity_score < 0.8);
    }

    #[test]
    fn test_check_continuity_exposure_mismatch() {
        let analyzer = ColorContinuityAnalyzer::default();
        let dark = make_color_frame(40, 40, 40, 50, 50);
        let bright = make_color_frame(220, 220, 220, 50, 50);
        let result = analyzer
            .check_continuity(&dark, &bright, 0, 1)
            .expect("should succeed in test");
        assert!(
            result.luminance_diff > 0.1,
            "dark vs bright should have high luminance diff"
        );
        let has_exposure_issue = result
            .issues
            .iter()
            .any(|i| i.category == ColorIssueCategory::Exposure);
        assert!(has_exposure_issue, "should detect exposure issue");
    }

    #[test]
    fn test_check_continuity_saturation_mismatch() {
        let analyzer = ColorContinuityAnalyzer::default();
        let saturated = make_color_frame(255, 0, 0, 50, 50);
        let desaturated = make_color_frame(128, 128, 128, 50, 50);
        let result = analyzer
            .check_continuity(&saturated, &desaturated, 0, 1)
            .expect("should succeed in test");
        assert!(result.saturation_diff > 0.05);
    }

    #[test]
    fn test_check_sequence_empty() {
        let analyzer = ColorContinuityAnalyzer::default();
        let results = analyzer
            .check_sequence(&[])
            .expect("should succeed in test");
        assert!(results.is_empty());
    }

    #[test]
    fn test_check_sequence_single() {
        let analyzer = ColorContinuityAnalyzer::default();
        let frames = vec![FrameBuffer::from_elem(50, 50, 3, 128)];
        let results = analyzer
            .check_sequence(&frames)
            .expect("should succeed in test");
        assert!(results.is_empty());
    }

    #[test]
    fn test_check_sequence_three_frames() {
        let analyzer = ColorContinuityAnalyzer::default();
        let frames = vec![
            make_color_frame(200, 180, 120, 50, 50), // warm
            make_color_frame(190, 175, 125, 50, 50), // slightly different warm
            make_color_frame(100, 150, 220, 50, 50), // cool
        ];
        let results = analyzer
            .check_sequence(&frames)
            .expect("should succeed in test");
        assert_eq!(results.len(), 2);
        // First pair should be more continuous than second
        assert!(results[0].continuity_score > results[1].continuity_score);
    }

    #[test]
    fn test_estimate_cct_daylight() {
        let cct = estimate_cct(200.0, 200.0, 200.0);
        // Neutral gray should be near D65 (6500K)
        assert!(
            cct > 5000.0 && cct < 8000.0,
            "neutral gray CCT should be near 6500K, got {cct}"
        );
    }

    #[test]
    fn test_estimate_cct_warm() {
        let cct_warm = estimate_cct(240.0, 180.0, 100.0);
        let cct_cool = estimate_cct(100.0, 150.0, 240.0);
        assert!(cct_warm < cct_cool, "warm should have lower CCT than cool");
    }

    #[test]
    fn test_estimate_cct_black() {
        let cct = estimate_cct(0.0, 0.0, 0.0);
        assert!((cct - 6500.0).abs() < 0.1, "black should default to 6500K");
    }

    #[test]
    fn test_estimate_tint_neutral() {
        let tint = estimate_tint(128.0, 128.0, 128.0);
        assert!(tint.abs() < 0.01, "neutral should have zero tint");
    }

    #[test]
    fn test_estimate_tint_green() {
        let tint = estimate_tint(100.0, 200.0, 100.0);
        assert!(tint > 0.0, "green-dominant should have positive tint");
    }

    #[test]
    fn test_estimate_tint_magenta() {
        let tint = estimate_tint(200.0, 100.0, 200.0);
        assert!(tint < 0.0, "magenta-dominant should have negative tint");
    }

    #[test]
    fn test_config_default() {
        let cfg = ColorContinuityConfig::default();
        assert!((cfg.max_temperature_diff_k - 500.0).abs() < f32::EPSILON);
        assert!((cfg.max_tint_diff - 0.05).abs() < f32::EPSILON);
    }

    #[test]
    fn test_config_accessor() {
        let cfg = ColorContinuityConfig {
            max_temperature_diff_k: 300.0,
            ..ColorContinuityConfig::default()
        };
        let analyzer = ColorContinuityAnalyzer::new(cfg);
        assert!((analyzer.config().max_temperature_diff_k - 300.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_issue_severity_ordering() {
        assert!(ColorIssueSeverity::Info < ColorIssueSeverity::Warning);
        assert!(ColorIssueSeverity::Warning < ColorIssueSeverity::Error);
    }

    #[test]
    fn test_issue_category_equality() {
        assert_eq!(
            ColorIssueCategory::Temperature,
            ColorIssueCategory::Temperature
        );
        assert_ne!(ColorIssueCategory::Temperature, ColorIssueCategory::Tint);
    }

    #[test]
    fn test_continuity_score_bounded() {
        let analyzer = ColorContinuityAnalyzer::default();
        let f1 = make_color_frame(50, 200, 50, 50, 50);
        let f2 = make_color_frame(200, 50, 200, 50, 50);
        let result = analyzer
            .check_continuity(&f1, &f2, 0, 1)
            .expect("should succeed in test");
        assert!(result.continuity_score >= 0.0 && result.continuity_score <= 1.0);
    }

    #[test]
    fn test_severe_mismatch_has_error_issues() {
        let cfg = ColorContinuityConfig {
            max_temperature_diff_k: 100.0,
            max_tint_diff: 0.01,
            max_luminance_diff: 0.01,
            max_saturation_diff: 0.01,
            max_channel_diff: 0.01,
        };
        let analyzer = ColorContinuityAnalyzer::new(cfg);
        let f1 = make_color_frame(255, 0, 0, 50, 50);
        let f2 = make_color_frame(0, 0, 255, 50, 50);
        let result = analyzer
            .check_continuity(&f1, &f2, 0, 1)
            .expect("should succeed in test");
        let has_error = result
            .issues
            .iter()
            .any(|i| i.severity == ColorIssueSeverity::Error);
        assert!(has_error, "extreme mismatch should have Error-level issues");
    }

    #[test]
    fn test_tint_diff_field() {
        let analyzer = ColorContinuityAnalyzer::default();
        let green_cast = make_color_frame(100, 200, 100, 50, 50);
        let magenta_cast = make_color_frame(200, 100, 200, 50, 50);
        let result = analyzer
            .check_continuity(&green_cast, &magenta_cast, 0, 1)
            .expect("should succeed in test");
        assert!(result.tint_diff > 0.0);
    }

    #[test]
    fn test_frame_color_stats_luminance_range() {
        let analyzer = ColorContinuityAnalyzer::default();
        let frame = make_color_frame(255, 255, 255, 50, 50);
        let stats = analyzer
            .compute_stats(&frame)
            .expect("should succeed in test");
        assert!(stats.luminance >= 0.0 && stats.luminance <= 1.0);
    }
}
