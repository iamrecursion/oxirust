#![allow(dead_code)]
//! Continuity checking for video scene sequences.
//!
//! Film and video continuity errors — mismatched props, costume changes
//! between cuts, lighting shifts, and colour-temperature jumps — are a
//! common quality issue.  This module provides automated heuristics to
//! flag potential continuity problems by comparing adjacent shots:
//!
//! - **Colour histogram consistency**: detect lighting / grade shifts.
//! - **Brightness consistency**: flag exposure jumps across cuts.
//! - **Edge structure matching**: detect object-level changes.
//! - **Aggregate scoring**: combine signals into a continuity score.

use std::fmt;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Kind of continuity issue detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContinuityIssue {
    /// Significant brightness jump between shots.
    BrightnessJump,
    /// Colour temperature / white-balance mismatch.
    ColourMismatch,
    /// Contrast level change.
    ContrastShift,
    /// Edge structure differs significantly (possible prop/costume change).
    EdgeStructureChange,
    /// Saturation shift between shots.
    SaturationShift,
}

impl fmt::Display for ContinuityIssue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BrightnessJump => write!(f, "BrightnessJump"),
            Self::ColourMismatch => write!(f, "ColourMismatch"),
            Self::ContrastShift => write!(f, "ContrastShift"),
            Self::EdgeStructureChange => write!(f, "EdgeStructureChange"),
            Self::SaturationShift => write!(f, "SaturationShift"),
        }
    }
}

/// A detected continuity problem between two shots.
#[derive(Debug, Clone)]
pub struct ContinuityAlert {
    /// Frame index of the first shot (last frame before cut).
    pub frame_a: usize,
    /// Frame index of the second shot (first frame after cut).
    pub frame_b: usize,
    /// Kind of issue.
    pub issue: ContinuityIssue,
    /// Magnitude of the difference (0..1).
    pub magnitude: f64,
    /// Severity label.
    pub severity: Severity,
}

/// Severity levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    /// Minor — possibly intentional.
    Low,
    /// Moderate — worth reviewing.
    Medium,
    /// Severe — likely a continuity error.
    High,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Low => write!(f, "Low"),
            Self::Medium => write!(f, "Medium"),
            Self::High => write!(f, "High"),
        }
    }
}

/// Configuration for continuity checking.
#[derive(Debug, Clone)]
pub struct ContinuityConfig {
    /// Brightness difference threshold to flag.
    pub brightness_threshold: f64,
    /// Colour histogram difference threshold.
    pub colour_threshold: f64,
    /// Contrast difference threshold.
    pub contrast_threshold: f64,
    /// Edge structure difference threshold.
    pub edge_threshold: f64,
    /// Saturation difference threshold.
    pub saturation_threshold: f64,
    /// Number of histogram bins.
    pub histogram_bins: usize,
}

/// Statistics for a single frame used in comparison.
#[derive(Debug, Clone)]
pub struct FrameStats {
    /// Frame index.
    pub frame_index: usize,
    /// Mean brightness (0..1).
    pub mean_brightness: f64,
    /// Brightness standard deviation.
    pub brightness_std: f64,
    /// Mean saturation (0..1, for greyscale this is 0).
    pub mean_saturation: f64,
    /// Normalised histogram (sums to 1).
    pub histogram: Vec<f64>,
    /// Mean edge magnitude.
    pub mean_edge: f64,
}

/// Aggregate continuity report for a sequence of shots.
#[derive(Debug, Clone)]
pub struct ContinuityReport {
    /// All detected alerts.
    pub alerts: Vec<ContinuityAlert>,
    /// Number of shot boundaries analysed.
    pub boundaries_checked: usize,
    /// Overall continuity score (0 = terrible, 1 = perfect).
    pub score: f64,
}

// ---------------------------------------------------------------------------
// Implementations
// ---------------------------------------------------------------------------

impl Default for ContinuityConfig {
    fn default() -> Self {
        Self {
            brightness_threshold: 0.12,
            colour_threshold: 0.20,
            contrast_threshold: 0.15,
            edge_threshold: 0.25,
            saturation_threshold: 0.15,
            histogram_bins: 32,
        }
    }
}

/// Compute frame statistics from a grayscale pixel buffer.
#[allow(clippy::cast_precision_loss)]
pub fn compute_frame_stats(pixels: &[f64], frame_index: usize, bins: usize) -> FrameStats {
    if pixels.is_empty() || bins == 0 {
        return FrameStats {
            frame_index,
            mean_brightness: 0.0,
            brightness_std: 0.0,
            mean_saturation: 0.0,
            histogram: Vec::new(),
            mean_edge: 0.0,
        };
    }

    let n = pixels.len() as f64;
    let mean = pixels.iter().sum::<f64>() / n;
    let var = pixels.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let std = var.sqrt();

    // Histogram
    let mut hist = vec![0.0f64; bins];
    for &v in pixels {
        let bin = ((v.clamp(0.0, 1.0)) * (bins as f64 - 1.0)).round() as usize;
        let bin = bin.min(bins - 1);
        hist[bin] += 1.0;
    }
    // Normalise
    let total: f64 = hist.iter().sum();
    if total > 0.0 {
        for h in &mut hist {
            *h /= total;
        }
    }

    // Simple edge metric: mean absolute difference of adjacent pixels
    let edge = if pixels.len() > 1 {
        pixels.windows(2).map(|w| (w[1] - w[0]).abs()).sum::<f64>() / (pixels.len() - 1) as f64
    } else {
        0.0
    };

    FrameStats {
        frame_index,
        mean_brightness: mean,
        brightness_std: std,
        mean_saturation: 0.0,
        histogram: hist,
        mean_edge: edge,
    }
}

/// Compare two frame stats and return continuity alerts.
fn compare_frames(
    a: &FrameStats,
    b: &FrameStats,
    config: &ContinuityConfig,
) -> Vec<ContinuityAlert> {
    let mut alerts = Vec::new();

    // Brightness
    let bdiff = (a.mean_brightness - b.mean_brightness).abs();
    if bdiff > config.brightness_threshold {
        alerts.push(ContinuityAlert {
            frame_a: a.frame_index,
            frame_b: b.frame_index,
            issue: ContinuityIssue::BrightnessJump,
            magnitude: bdiff.min(1.0),
            severity: severity_from_magnitude(bdiff, config.brightness_threshold),
        });
    }

    // Contrast
    let cdiff = (a.brightness_std - b.brightness_std).abs();
    if cdiff > config.contrast_threshold {
        alerts.push(ContinuityAlert {
            frame_a: a.frame_index,
            frame_b: b.frame_index,
            issue: ContinuityIssue::ContrastShift,
            magnitude: cdiff.min(1.0),
            severity: severity_from_magnitude(cdiff, config.contrast_threshold),
        });
    }

    // Colour histogram (chi-squared distance)
    if a.histogram.len() == b.histogram.len() && !a.histogram.is_empty() {
        let chi2: f64 = a
            .histogram
            .iter()
            .zip(b.histogram.iter())
            .map(|(ha, hb)| {
                let sum = ha + hb;
                if sum > 0.0 {
                    (ha - hb).powi(2) / sum
                } else {
                    0.0
                }
            })
            .sum();
        if chi2 > config.colour_threshold {
            alerts.push(ContinuityAlert {
                frame_a: a.frame_index,
                frame_b: b.frame_index,
                issue: ContinuityIssue::ColourMismatch,
                magnitude: chi2.min(1.0),
                severity: severity_from_magnitude(chi2, config.colour_threshold),
            });
        }
    }

    // Edge structure
    let ediff = (a.mean_edge - b.mean_edge).abs();
    if ediff > config.edge_threshold {
        alerts.push(ContinuityAlert {
            frame_a: a.frame_index,
            frame_b: b.frame_index,
            issue: ContinuityIssue::EdgeStructureChange,
            magnitude: ediff.min(1.0),
            severity: severity_from_magnitude(ediff, config.edge_threshold),
        });
    }

    // Saturation
    let sdiff = (a.mean_saturation - b.mean_saturation).abs();
    if sdiff > config.saturation_threshold {
        alerts.push(ContinuityAlert {
            frame_a: a.frame_index,
            frame_b: b.frame_index,
            issue: ContinuityIssue::SaturationShift,
            magnitude: sdiff.min(1.0),
            severity: severity_from_magnitude(sdiff, config.saturation_threshold),
        });
    }

    alerts
}

/// Map a magnitude and threshold to a severity level.
fn severity_from_magnitude(mag: f64, threshold: f64) -> Severity {
    let ratio = mag / threshold;
    if ratio > 3.0 {
        Severity::High
    } else if ratio > 1.5 {
        Severity::Medium
    } else {
        Severity::Low
    }
}

/// Run continuity checks across a sequence of frame statistics.
#[allow(clippy::cast_precision_loss)]
pub fn check_continuity(frame_stats: &[FrameStats], config: &ContinuityConfig) -> ContinuityReport {
    let mut all_alerts = Vec::new();
    let boundaries = if frame_stats.len() > 1 {
        frame_stats.len() - 1
    } else {
        0
    };

    for pair in frame_stats.windows(2) {
        let alerts = compare_frames(&pair[0], &pair[1], config);
        all_alerts.extend(alerts);
    }

    let score = if boundaries > 0 {
        let penalty: f64 = all_alerts.iter().map(|a| a.magnitude).sum();
        (1.0 - penalty / boundaries as f64).clamp(0.0, 1.0)
    } else {
        1.0
    };

    ContinuityReport {
        alerts: all_alerts,
        boundaries_checked: boundaries,
        score,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_frame_stats_empty() {
        let fs = compute_frame_stats(&[], 0, 16);
        assert_eq!(fs.mean_brightness, 0.0);
        assert!(fs.histogram.is_empty());
    }

    #[test]
    fn test_compute_frame_stats_uniform() {
        let px = vec![0.5; 100];
        let fs = compute_frame_stats(&px, 0, 16);
        assert!((fs.mean_brightness - 0.5).abs() < 1e-9);
        assert!(fs.brightness_std < 1e-9);
    }

    #[test]
    fn test_histogram_sums_to_one() {
        let px: Vec<f64> = (0..100).map(|i| i as f64 / 99.0).collect();
        let fs = compute_frame_stats(&px, 0, 10);
        let total: f64 = fs.histogram.iter().sum();
        assert!((total - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_compare_identical_no_alerts() {
        let px = vec![0.5; 100];
        let a = compute_frame_stats(&px, 0, 16);
        let b = compute_frame_stats(&px, 1, 16);
        let alerts = compare_frames(&a, &b, &ContinuityConfig::default());
        assert!(alerts.is_empty());
    }

    #[test]
    fn test_brightness_jump_detected() {
        let a = compute_frame_stats(&vec![0.2; 100], 0, 16);
        let b = compute_frame_stats(&vec![0.8; 100], 1, 16);
        let alerts = compare_frames(&a, &b, &ContinuityConfig::default());
        assert!(alerts
            .iter()
            .any(|al| al.issue == ContinuityIssue::BrightnessJump));
    }

    #[test]
    fn test_severity_levels() {
        assert_eq!(severity_from_magnitude(0.13, 0.12), Severity::Low);
        assert_eq!(severity_from_magnitude(0.25, 0.12), Severity::Medium);
        assert_eq!(severity_from_magnitude(0.50, 0.12), Severity::High);
    }

    #[test]
    fn test_check_continuity_empty() {
        let report = check_continuity(&[], &ContinuityConfig::default());
        assert_eq!(report.boundaries_checked, 0);
        assert_eq!(report.score, 1.0);
    }

    #[test]
    fn test_check_continuity_single_frame() {
        let fs = vec![compute_frame_stats(&vec![0.5; 100], 0, 16)];
        let report = check_continuity(&fs, &ContinuityConfig::default());
        assert_eq!(report.boundaries_checked, 0);
    }

    #[test]
    fn test_check_continuity_clean_sequence() {
        let stats: Vec<FrameStats> = (0..5)
            .map(|i| compute_frame_stats(&vec![0.5; 100], i, 16))
            .collect();
        let report = check_continuity(&stats, &ContinuityConfig::default());
        assert!(report.alerts.is_empty());
        assert!((report.score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_check_continuity_with_jump() {
        let mut stats: Vec<FrameStats> = (0..3)
            .map(|i| compute_frame_stats(&vec![0.3; 100], i, 16))
            .collect();
        stats.push(compute_frame_stats(&vec![0.9; 100], 3, 16));
        let report = check_continuity(&stats, &ContinuityConfig::default());
        assert!(!report.alerts.is_empty());
        assert!(report.score < 1.0);
    }

    #[test]
    fn test_issue_display() {
        assert_eq!(
            format!("{}", ContinuityIssue::BrightnessJump),
            "BrightnessJump"
        );
        assert_eq!(
            format!("{}", ContinuityIssue::ColourMismatch),
            "ColourMismatch"
        );
    }

    #[test]
    fn test_severity_display() {
        assert_eq!(format!("{}", Severity::Low), "Low");
        assert_eq!(format!("{}", Severity::High), "High");
    }

    #[test]
    fn test_default_config() {
        let c = ContinuityConfig::default();
        assert_eq!(c.histogram_bins, 32);
        assert!(c.brightness_threshold > 0.0);
    }

    #[test]
    fn test_edge_metric() {
        let px = vec![0.0, 1.0, 0.0, 1.0];
        let fs = compute_frame_stats(&px, 0, 8);
        assert!(fs.mean_edge > 0.5);
    }
}
