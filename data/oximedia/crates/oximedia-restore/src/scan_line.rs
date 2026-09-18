#![allow(dead_code)]
//! Scan-line artefact detection and repair for interlaced and analogue sources.
//!
//! Analogue video, VHS tapes, and CRT captures frequently exhibit scan-line
//! artefacts — visible horizontal lines caused by interlacing, head-switching,
//! or signal degradation.  This module provides detection, measurement, and
//! correction tools:
//!
//! - **Scan-line detection**: identify rows affected by artefacts.
//! - **Interpolation repair**: replace bad lines via neighbour interpolation.
//! - **De-interlacing helpers**: field separation and weaving utilities.

use std::fmt;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Kind of scan-line artefact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanLineArtifact {
    /// Missing or black line.
    Dropout,
    /// Noisy / corrupted line.
    Noise,
    /// Shifted horizontally relative to neighbours.
    HorizontalShift,
    /// Brightness spike compared to neighbours.
    BrightnessSpike,
}

impl fmt::Display for ScanLineArtifact {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dropout => write!(f, "Dropout"),
            Self::Noise => write!(f, "Noise"),
            Self::HorizontalShift => write!(f, "HorizontalShift"),
            Self::BrightnessSpike => write!(f, "BrightnessSpike"),
        }
    }
}

/// Description of a single detected artefact line.
#[derive(Debug, Clone)]
pub struct DetectedLine {
    /// Row index in the frame.
    pub row: usize,
    /// Kind of artefact.
    pub artifact: ScanLineArtifact,
    /// Confidence score (0..1).
    pub confidence: f64,
    /// Severity of the artefact (0..1).
    pub severity: f64,
}

/// Configuration for the scan-line detector.
#[derive(Debug, Clone)]
pub struct ScanLineDetectorConfig {
    /// Minimum row-to-row brightness difference to flag as artefact.
    pub brightness_threshold: f64,
    /// Minimum correlation drop to flag noise.
    pub correlation_threshold: f64,
    /// Whether to check for horizontal shift.
    pub detect_shift: bool,
    /// Maximum horizontal shift in pixels to test.
    pub max_shift_pixels: usize,
}

/// Interpolation method for scan-line repair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationMethod {
    /// Simple average of row above and below.
    Linear,
    /// Cubic interpolation using 4 neighbours.
    Cubic,
    /// Copy from the nearest valid neighbour.
    NearestNeighbour,
}

/// Result of a scan-line repair pass.
#[derive(Debug, Clone)]
pub struct RepairResult {
    /// Repaired pixel data (row-major, grayscale).
    pub pixels: Vec<f64>,
    /// Width.
    pub width: usize,
    /// Height.
    pub height: usize,
    /// Number of lines repaired.
    pub lines_repaired: usize,
}

/// Field parity for interlaced content.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldParity {
    /// Upper field first (even rows).
    Upper,
    /// Lower field first (odd rows).
    Lower,
}

/// Separated fields from an interlaced frame.
#[derive(Debug, Clone)]
pub struct SeparatedFields {
    /// Even rows.
    pub upper: Vec<f64>,
    /// Odd rows.
    pub lower: Vec<f64>,
    /// Width.
    pub width: usize,
    /// Height of each field.
    pub field_height: usize,
}

// ---------------------------------------------------------------------------
// Implementations
// ---------------------------------------------------------------------------

impl Default for ScanLineDetectorConfig {
    fn default() -> Self {
        Self {
            brightness_threshold: 0.15,
            correlation_threshold: 0.7,
            detect_shift: true,
            max_shift_pixels: 4,
        }
    }
}

/// Compute mean of a row.
#[allow(clippy::cast_precision_loss)]
fn row_mean(pixels: &[f64], width: usize, row: usize) -> f64 {
    if width == 0 {
        return 0.0;
    }
    let start = row * width;
    let end = start + width;
    if end > pixels.len() {
        return 0.0;
    }
    pixels[start..end].iter().sum::<f64>() / width as f64
}

/// Compute correlation between two rows.
#[allow(clippy::cast_precision_loss)]
fn row_correlation(pixels: &[f64], width: usize, r1: usize, r2: usize) -> f64 {
    if width == 0 {
        return 0.0;
    }
    let s1 = r1 * width;
    let s2 = r2 * width;
    if s1 + width > pixels.len() || s2 + width > pixels.len() {
        return 0.0;
    }
    let m1 = row_mean(pixels, width, r1);
    let m2 = row_mean(pixels, width, r2);
    let mut cov = 0.0f64;
    let mut v1 = 0.0f64;
    let mut v2 = 0.0f64;
    for i in 0..width {
        let a = pixels[s1 + i] - m1;
        let b = pixels[s2 + i] - m2;
        cov += a * b;
        v1 += a * a;
        v2 += b * b;
    }
    let denom = (v1 * v2).sqrt();
    if denom < 1e-12 {
        return 1.0; // identical rows
    }
    cov / denom
}

/// Detect scan-line artefacts in a grayscale image.
pub fn detect_scan_lines(
    pixels: &[f64],
    width: usize,
    height: usize,
    config: &ScanLineDetectorConfig,
) -> Vec<DetectedLine> {
    let mut results = Vec::new();
    if width == 0 || height < 3 || pixels.len() < width * height {
        return results;
    }

    for row in 1..height - 1 {
        let mean_prev = row_mean(pixels, width, row - 1);
        let mean_curr = row_mean(pixels, width, row);
        let mean_next = row_mean(pixels, width, row + 1);
        let expected = (mean_prev + mean_next) / 2.0;
        let diff = (mean_curr - expected).abs();

        // Brightness spike / dropout
        if diff > config.brightness_threshold {
            let severity = (diff / config.brightness_threshold).min(1.0);
            let artifact = if mean_curr < expected * 0.5 {
                ScanLineArtifact::Dropout
            } else {
                ScanLineArtifact::BrightnessSpike
            };
            results.push(DetectedLine {
                row,
                artifact,
                confidence: severity,
                severity,
            });
            continue;
        }

        // Correlation-based noise detection
        let corr_prev = row_correlation(pixels, width, row, row - 1);
        let corr_next = row_correlation(pixels, width, row, row + 1);
        let avg_corr = (corr_prev + corr_next) / 2.0;
        if avg_corr < config.correlation_threshold {
            let severity = (1.0 - avg_corr).min(1.0);
            results.push(DetectedLine {
                row,
                artifact: ScanLineArtifact::Noise,
                confidence: severity,
                severity,
            });
        }
    }
    results
}

/// Repair detected scan lines using interpolation.
pub fn repair_scan_lines(
    pixels: &[f64],
    width: usize,
    height: usize,
    bad_rows: &[usize],
    method: InterpolationMethod,
) -> RepairResult {
    let mut output = pixels.to_vec();
    let mut repaired = 0usize;

    for &row in bad_rows {
        if row == 0 || row >= height - 1 {
            continue;
        }
        let prev_start = (row - 1) * width;
        let curr_start = row * width;
        let next_start = (row + 1) * width;

        if curr_start + width > output.len() || next_start + width > output.len() {
            continue;
        }

        match method {
            InterpolationMethod::Linear => {
                for i in 0..width {
                    output[curr_start + i] =
                        (output[prev_start + i] + output[next_start + i]) / 2.0;
                }
            }
            InterpolationMethod::Cubic => {
                // Use 4-row cubic if available, else fallback to linear
                if row >= 2 && row + 2 < height {
                    let pp_start = (row - 2) * width;
                    let nn_start = (row + 2) * width;
                    for i in 0..width {
                        let a = output[pp_start + i];
                        let b = output[prev_start + i];
                        let d = output[next_start + i];
                        let e = if nn_start + i < output.len() {
                            output[nn_start + i]
                        } else {
                            d
                        };
                        output[curr_start + i] = (-a + 9.0 * b + 9.0 * d - e) / 16.0;
                    }
                } else {
                    for i in 0..width {
                        output[curr_start + i] =
                            (output[prev_start + i] + output[next_start + i]) / 2.0;
                    }
                }
            }
            InterpolationMethod::NearestNeighbour => {
                for i in 0..width {
                    output[curr_start + i] = output[prev_start + i];
                }
            }
        }
        repaired += 1;
    }

    RepairResult {
        pixels: output,
        width,
        height,
        lines_repaired: repaired,
    }
}

/// Separate interlaced frame into two fields.
pub fn separate_fields(pixels: &[f64], width: usize, height: usize) -> SeparatedFields {
    let mut upper = Vec::new();
    let mut lower = Vec::new();
    for row in 0..height {
        let start = row * width;
        let end = (start + width).min(pixels.len());
        if start >= pixels.len() {
            break;
        }
        if row % 2 == 0 {
            upper.extend_from_slice(&pixels[start..end]);
        } else {
            lower.extend_from_slice(&pixels[start..end]);
        }
    }
    let field_height = (height + 1) / 2;
    SeparatedFields {
        upper,
        lower,
        width,
        field_height,
    }
}

/// Weave two fields back into a single frame.
pub fn weave_fields(fields: &SeparatedFields) -> Vec<f64> {
    let height = fields.field_height * 2;
    let total = height * fields.width;
    let mut output = vec![0.0; total];
    let w = fields.width;

    for i in 0..fields.field_height {
        let upper_start = i * w;
        let lower_start = i * w;
        let frame_upper_row = i * 2;
        let frame_lower_row = i * 2 + 1;

        for col in 0..w {
            if upper_start + col < fields.upper.len() {
                output[frame_upper_row * w + col] = fields.upper[upper_start + col];
            }
            if lower_start + col < fields.lower.len() && frame_lower_row * w + col < total {
                output[frame_lower_row * w + col] = fields.lower[lower_start + col];
            }
        }
    }
    output
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(width: usize, height: usize, value: f64) -> Vec<f64> {
        vec![value; width * height]
    }

    #[test]
    fn test_row_mean() {
        let px = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert!((row_mean(&px, 3, 0) - 2.0).abs() < 1e-9);
        assert!((row_mean(&px, 3, 1) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_row_correlation_identical() {
        let px = vec![0.1, 0.2, 0.3, 0.1, 0.2, 0.3];
        let c = row_correlation(&px, 3, 0, 1);
        assert!((c - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_detect_empty() {
        let lines = detect_scan_lines(&[], 0, 0, &ScanLineDetectorConfig::default());
        assert!(lines.is_empty());
    }

    #[test]
    fn test_detect_uniform_no_artifacts() {
        let px = make_frame(10, 10, 0.5);
        let lines = detect_scan_lines(&px, 10, 10, &ScanLineDetectorConfig::default());
        assert!(lines.is_empty());
    }

    #[test]
    fn test_detect_dropout() {
        let mut px = make_frame(10, 10, 0.5);
        // Make row 5 all black
        for i in 0..10 {
            px[5 * 10 + i] = 0.0;
        }
        let lines = detect_scan_lines(&px, 10, 10, &ScanLineDetectorConfig::default());
        assert!(!lines.is_empty());
        assert!(lines.iter().any(|l| l.row == 5));
    }

    #[test]
    fn test_detect_brightness_spike() {
        let mut px = make_frame(10, 10, 0.3);
        for i in 0..10 {
            px[4 * 10 + i] = 1.0;
        }
        let lines = detect_scan_lines(&px, 10, 10, &ScanLineDetectorConfig::default());
        assert!(lines
            .iter()
            .any(|l| l.row == 4 && l.artifact == ScanLineArtifact::BrightnessSpike));
    }

    #[test]
    fn test_repair_linear() {
        let mut px = make_frame(4, 5, 0.5);
        px[2 * 4..3 * 4].fill(0.0); // row 2 is bad
        let result = repair_scan_lines(&px, 4, 5, &[2], InterpolationMethod::Linear);
        assert_eq!(result.lines_repaired, 1);
        for i in 0..4 {
            assert!((result.pixels[2 * 4 + i] - 0.5).abs() < 1e-9);
        }
    }

    #[test]
    fn test_repair_cubic() {
        let mut px = make_frame(4, 6, 0.6);
        px[3 * 4..4 * 4].fill(0.0);
        let result = repair_scan_lines(&px, 4, 6, &[3], InterpolationMethod::Cubic);
        assert_eq!(result.lines_repaired, 1);
    }

    #[test]
    fn test_repair_nearest() {
        let mut px = make_frame(4, 5, 0.4);
        px[2 * 4..3 * 4].fill(0.0);
        let result = repair_scan_lines(&px, 4, 5, &[2], InterpolationMethod::NearestNeighbour);
        assert_eq!(result.lines_repaired, 1);
        for i in 0..4 {
            assert!((result.pixels[2 * 4 + i] - 0.4).abs() < 1e-9);
        }
    }

    #[test]
    fn test_separate_fields_sizes() {
        let px = make_frame(4, 8, 0.5);
        let fields = separate_fields(&px, 4, 8);
        assert_eq!(fields.upper.len(), 4 * 4); // 4 even rows
        assert_eq!(fields.lower.len(), 4 * 4); // 4 odd rows
        assert_eq!(fields.field_height, 4);
    }

    #[test]
    fn test_weave_roundtrip() {
        let px = (0..32).map(|i| i as f64 / 32.0).collect::<Vec<_>>();
        let fields = separate_fields(&px, 4, 8);
        let woven = weave_fields(&fields);
        assert_eq!(woven.len(), 32);
        for (a, b) in px.iter().zip(woven.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn test_artifact_display() {
        assert_eq!(format!("{}", ScanLineArtifact::Dropout), "Dropout");
        assert_eq!(format!("{}", ScanLineArtifact::Noise), "Noise");
    }

    #[test]
    fn test_default_config() {
        let c = ScanLineDetectorConfig::default();
        assert!(c.brightness_threshold > 0.0);
        assert!(c.detect_shift);
    }

    #[test]
    fn test_repair_boundary_rows_skipped() {
        let px = make_frame(4, 4, 0.5);
        let result = repair_scan_lines(&px, 4, 4, &[0, 3], InterpolationMethod::Linear);
        assert_eq!(result.lines_repaired, 0);
    }
}
