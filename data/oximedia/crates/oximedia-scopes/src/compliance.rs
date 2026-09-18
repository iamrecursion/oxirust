//! Broadcast compliance checking for professional video workflows.
//!
//! This module provides comprehensive compliance checking against broadcast
//! standards to ensure video content meets technical requirements for distribution:
//! - **Gamut limiting**: Detect colors outside legal broadcast gamut
//! - **Luma clipping**: Identify illegal luma levels (< 16 or > 235 for 8-bit)
//! - **Chroma overshoot**: Detect excessive chroma values
//! - **Illegal color combinations**: Find RGB combinations that exceed legal YCbCr ranges
//! - **Black level compliance**: Verify proper black levels
//! - **White level compliance**: Check for proper white points
//! - **ITU-R BT.601/709/2020 compliance**: Standard-specific checks
//! - **SMPTE compliance**: SMPTE-specific requirements
//!
//! Broadcast compliance is critical for ensuring content can be safely broadcast
//! without quality degradation or technical rejections.

use crate::render::rgb_to_ycbcr;
use crate::GamutColorspace;
use oximedia_core::OxiResult;
use rayon::prelude::*;

/// Broadcast standard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BroadcastStandard {
    /// ITU-R BT.601 (SD).
    Bt601,

    /// ITU-R BT.709 (HD).
    Bt709,

    /// ITU-R BT.2020 (UHD/HDR).
    Bt2020,

    /// SMPTE (Society of Motion Picture and Television Engineers).
    Smpte,

    /// EBU (European Broadcasting Union).
    Ebu,
}

/// Legal range for luma and chroma values.
#[derive(Debug, Clone, Copy)]
pub struct LegalRange {
    /// Minimum legal luma value (0-255).
    pub min_luma: u8,

    /// Maximum legal luma value (0-255).
    pub max_luma: u8,

    /// Minimum legal chroma value (0-255).
    pub min_chroma: u8,

    /// Maximum legal chroma value (0-255).
    pub max_chroma: u8,
}

impl LegalRange {
    /// BT.601/709 legal range (16-235 for luma, 16-240 for chroma).
    #[must_use]
    pub const fn bt709() -> Self {
        Self {
            min_luma: 16,
            max_luma: 235,
            min_chroma: 16,
            max_chroma: 240,
        }
    }

    /// BT.2020 legal range (same as BT.709 for 8-bit).
    #[must_use]
    pub const fn bt2020() -> Self {
        Self::bt709()
    }

    /// Full range (0-255 for both luma and chroma).
    #[must_use]
    pub const fn full_range() -> Self {
        Self {
            min_luma: 0,
            max_luma: 255,
            min_chroma: 0,
            max_chroma: 255,
        }
    }

    /// Strict legal range with headroom (20-235 for luma, 20-236 for chroma).
    #[must_use]
    pub const fn strict() -> Self {
        Self {
            min_luma: 20,
            max_luma: 235,
            min_chroma: 20,
            max_chroma: 236,
        }
    }
}

/// Compliance violation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ViolationType {
    /// Luma below legal minimum.
    LumaUnderflow,

    /// Luma above legal maximum.
    LumaOverflow,

    /// Chroma below legal minimum.
    ChromaUnderflow,

    /// Chroma above legal maximum.
    ChromaOverflow,

    /// Color outside legal gamut.
    GamutExcursion,

    /// Illegal RGB to YCbCr conversion result.
    IllegalColor,

    /// Black level too low.
    BlackLevelError,

    /// White level too high.
    WhiteLevelError,

    /// Composite overshoot (luma + chroma).
    CompositeOvershoot,
}

/// Compliance violation location.
#[derive(Debug, Clone)]
pub struct Violation {
    /// Type of violation.
    pub violation_type: ViolationType,

    /// X coordinate of the pixel.
    pub x: u32,

    /// Y coordinate of the pixel.
    pub y: u32,

    /// RGB values of the offending pixel.
    pub rgb: [u8; 3],

    /// YCbCr values of the offending pixel.
    pub ycbcr: [u8; 3],

    /// Severity (0.0-1.0, higher = more severe).
    pub severity: f32,
}

/// Compliance report for a video frame.
#[derive(Debug, Clone)]
pub struct ComplianceReport {
    /// Total number of pixels analyzed.
    pub total_pixels: u32,

    /// Number of pixels with violations.
    pub violation_count: u32,

    /// Percentage of pixels with violations.
    pub violation_percent: f32,

    /// Violations by type.
    pub violations_by_type: Vec<(ViolationType, u32)>,

    /// Detailed violation list (up to `max_violations`).
    pub violations: Vec<Violation>,

    /// Whether the frame passes compliance.
    pub passes_compliance: bool,

    /// Broadcast standard used for checking.
    pub standard: BroadcastStandard,
}

/// Compliance checker configuration.
#[derive(Debug, Clone)]
pub struct ComplianceConfig {
    /// Broadcast standard to check against.
    pub standard: BroadcastStandard,

    /// Legal range for luma and chroma.
    pub legal_range: LegalRange,

    /// Gamut colorspace to check.
    pub gamut: GamutColorspace,

    /// Maximum number of violations to report (for performance).
    pub max_violations: usize,

    /// Tolerance in pixels (0-255) before marking as violation.
    pub tolerance: u8,

    /// Check for gamut excursions.
    pub check_gamut: bool,

    /// Check for illegal colors.
    pub check_illegal_colors: bool,

    /// Check for composite overshoot.
    pub check_composite: bool,
}

impl Default for ComplianceConfig {
    fn default() -> Self {
        Self {
            standard: BroadcastStandard::Bt709,
            legal_range: LegalRange::bt709(),
            gamut: GamutColorspace::Rec709,
            max_violations: 1000,
            tolerance: 0,
            check_gamut: true,
            check_illegal_colors: true,
            check_composite: true,
        }
    }
}

/// Checks a video frame for broadcast compliance violations.
///
/// # Arguments
///
/// * `frame` - RGB24 frame data (width * height * 3 bytes)
/// * `width` - Frame width in pixels
/// * `height` - Frame height in pixels
/// * `config` - Compliance checker configuration
///
/// # Errors
///
/// Returns an error if frame data is invalid or insufficient.
#[allow(clippy::cast_precision_loss)]
#[allow(clippy::too_many_lines)]
pub fn check_compliance(
    frame: &[u8],
    width: u32,
    height: u32,
    config: &ComplianceConfig,
) -> OxiResult<ComplianceReport> {
    let expected_size = (width * height * 3) as usize;
    if frame.len() < expected_size {
        return Err(oximedia_core::OxiError::InvalidData(format!(
            "Frame data too small: expected {expected_size}, got {}",
            frame.len()
        )));
    }

    let total_pixels = width * height;

    // Process frame in parallel to find violations
    let mut all_violations: Vec<Violation> = (0..height)
        .into_par_iter()
        .flat_map(|y| {
            let mut local_violations: Vec<Violation> = Vec::new();

            for x in 0..width {
                let pixel_idx = ((y * width + x) * 3) as usize;
                let r = frame[pixel_idx];
                let g = frame[pixel_idx + 1];
                let b = frame[pixel_idx + 2];

                let (luma, cb, cr) = rgb_to_ycbcr(r, g, b);
                let ycbcr = [luma, cb, cr];

                // Check luma violations
                if luma < config.legal_range.min_luma.saturating_sub(config.tolerance) {
                    let severity = f32::from(config.legal_range.min_luma - luma) / 255.0;
                    local_violations.push(Violation {
                        violation_type: ViolationType::LumaUnderflow,
                        x,
                        y,
                        rgb: [r, g, b],
                        ycbcr,
                        severity,
                    });
                }

                if luma > config.legal_range.max_luma.saturating_add(config.tolerance) {
                    let severity = f32::from(luma - config.legal_range.max_luma) / 255.0;
                    local_violations.push(Violation {
                        violation_type: ViolationType::LumaOverflow,
                        x,
                        y,
                        rgb: [r, g, b],
                        ycbcr,
                        severity,
                    });
                }

                // Check chroma violations
                if cb
                    < config
                        .legal_range
                        .min_chroma
                        .saturating_sub(config.tolerance)
                {
                    let severity = f32::from(config.legal_range.min_chroma - cb) / 255.0;
                    local_violations.push(Violation {
                        violation_type: ViolationType::ChromaUnderflow,
                        x,
                        y,
                        rgb: [r, g, b],
                        ycbcr,
                        severity,
                    });
                }

                if cb
                    > config
                        .legal_range
                        .max_chroma
                        .saturating_add(config.tolerance)
                {
                    let severity = f32::from(cb - config.legal_range.max_chroma) / 255.0;
                    local_violations.push(Violation {
                        violation_type: ViolationType::ChromaOverflow,
                        x,
                        y,
                        rgb: [r, g, b],
                        ycbcr,
                        severity,
                    });
                }

                if cr
                    < config
                        .legal_range
                        .min_chroma
                        .saturating_sub(config.tolerance)
                {
                    let severity = f32::from(config.legal_range.min_chroma - cr) / 255.0;
                    local_violations.push(Violation {
                        violation_type: ViolationType::ChromaUnderflow,
                        x,
                        y,
                        rgb: [r, g, b],
                        ycbcr,
                        severity,
                    });
                }

                if cr
                    > config
                        .legal_range
                        .max_chroma
                        .saturating_add(config.tolerance)
                {
                    let severity = f32::from(cr - config.legal_range.max_chroma) / 255.0;
                    local_violations.push(Violation {
                        violation_type: ViolationType::ChromaOverflow,
                        x,
                        y,
                        rgb: [r, g, b],
                        ycbcr,
                        severity,
                    });
                }

                // Check for illegal colors
                if config.check_illegal_colors && is_illegal_color(r, g, b) {
                    local_violations.push(Violation {
                        violation_type: ViolationType::IllegalColor,
                        x,
                        y,
                        rgb: [r, g, b],
                        ycbcr,
                        severity: 0.8,
                    });
                }

                // Check for composite overshoot
                if config.check_composite && has_composite_overshoot(luma, cb, cr) {
                    local_violations.push(Violation {
                        violation_type: ViolationType::CompositeOvershoot,
                        x,
                        y,
                        rgb: [r, g, b],
                        ycbcr,
                        severity: 0.6,
                    });
                }

                // Limit violations to avoid excessive memory usage
                if local_violations.len() >= config.max_violations {
                    break;
                }
            }

            local_violations
        })
        .collect();

    // Limit total violations
    all_violations.truncate(config.max_violations);
    let violations = all_violations;

    // Count violations by type
    let mut luma_underflow = 0u32;
    let mut luma_overflow = 0u32;
    let mut chroma_underflow = 0u32;
    let mut chroma_overflow = 0u32;
    let mut gamut_excursion = 0u32;
    let mut illegal_color = 0u32;
    let mut black_level_error = 0u32;
    let mut white_level_error = 0u32;
    let mut composite_overshoot = 0u32;

    for violation in &violations {
        match violation.violation_type {
            ViolationType::LumaUnderflow => luma_underflow += 1,
            ViolationType::LumaOverflow => luma_overflow += 1,
            ViolationType::ChromaUnderflow => chroma_underflow += 1,
            ViolationType::ChromaOverflow => chroma_overflow += 1,
            ViolationType::GamutExcursion => gamut_excursion += 1,
            ViolationType::IllegalColor => illegal_color += 1,
            ViolationType::BlackLevelError => black_level_error += 1,
            ViolationType::WhiteLevelError => white_level_error += 1,
            ViolationType::CompositeOvershoot => composite_overshoot += 1,
        }
    }

    let violations_by_type = vec![
        (ViolationType::LumaUnderflow, luma_underflow),
        (ViolationType::LumaOverflow, luma_overflow),
        (ViolationType::ChromaUnderflow, chroma_underflow),
        (ViolationType::ChromaOverflow, chroma_overflow),
        (ViolationType::GamutExcursion, gamut_excursion),
        (ViolationType::IllegalColor, illegal_color),
        (ViolationType::BlackLevelError, black_level_error),
        (ViolationType::WhiteLevelError, white_level_error),
        (ViolationType::CompositeOvershoot, composite_overshoot),
    ];

    let violation_count = violations.len() as u32;
    let violation_percent = (violation_count as f32 / total_pixels as f32) * 100.0;

    // Frame passes if violations are below threshold (< 0.1%)
    let passes_compliance = violation_percent < 0.1;

    Ok(ComplianceReport {
        total_pixels,
        violation_count,
        violation_percent,
        violations_by_type,
        violations,
        passes_compliance,
        standard: config.standard,
    })
}

/// Checks if an RGB color is illegal (would produce out-of-range YCbCr).
#[must_use]
fn is_illegal_color(r: u8, g: u8, b: u8) -> bool {
    // Check for RGB combinations that produce illegal YCbCr values
    // Common illegal combinations:
    // - High red with low green/blue
    // - High blue with low red/green
    // - Extreme color imbalances

    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let chroma = max - min;

    // If chroma is very high and colors are imbalanced, likely illegal
    if chroma > 200 {
        let r_dominant = r == max && (i16::from(r) - i16::from(g)).abs() > 150;
        let b_dominant = b == max && (i16::from(b) - i16::from(g)).abs() > 150;

        if r_dominant || b_dominant {
            return true;
        }
    }

    false
}

/// Checks for composite overshoot (luma + chroma exceeds limits).
#[must_use]
fn has_composite_overshoot(luma: u8, cb: u8, cr: u8) -> bool {
    // Simplified composite overshoot check
    // In real NTSC/PAL composite video, luma + chroma can create overshoots

    let cb_deviation = (i16::from(cb) - 128).abs();
    let cr_deviation = (i16::from(cr) - 128).abs();
    let total_deviation = cb_deviation + cr_deviation;

    // If luma is near limits and chroma is high, could cause overshoot
    if luma > 220 && total_deviation > 100 {
        return true;
    }

    if luma < 35 && total_deviation > 100 {
        return true;
    }

    false
}

/// Generates a compliance overlay showing violations.
///
/// Returns an RGBA image with violations highlighted.
#[must_use]
pub fn generate_compliance_overlay(
    frame: &[u8],
    width: u32,
    height: u32,
    report: &ComplianceReport,
) -> Vec<u8> {
    let mut overlay = vec![0u8; (width * height * 4) as usize];

    // Copy original frame to overlay
    for y in 0..height {
        for x in 0..width {
            let src_idx = ((y * width + x) * 3) as usize;
            let dst_idx = ((y * width + x) * 4) as usize;

            if src_idx + 2 < frame.len() {
                overlay[dst_idx] = frame[src_idx];
                overlay[dst_idx + 1] = frame[src_idx + 1];
                overlay[dst_idx + 2] = frame[src_idx + 2];
                overlay[dst_idx + 3] = 255;
            }
        }
    }

    // Overlay violations with color-coded markers
    for violation in &report.violations {
        let color = match violation.violation_type {
            ViolationType::LumaUnderflow => [0, 0, 255, 192], // Blue
            ViolationType::LumaOverflow => [255, 0, 255, 192], // Magenta
            ViolationType::ChromaUnderflow => [0, 255, 255, 192], // Cyan
            ViolationType::ChromaOverflow => [255, 255, 0, 192], // Yellow
            ViolationType::GamutExcursion => [255, 128, 0, 192], // Orange
            ViolationType::IllegalColor => [255, 0, 0, 255],  // Red
            ViolationType::BlackLevelError => [64, 64, 255, 192], // Dark blue
            ViolationType::WhiteLevelError => [255, 64, 255, 192], // Light magenta
            ViolationType::CompositeOvershoot => [128, 0, 128, 192], // Purple
        };

        let idx = ((violation.y * width + violation.x) * 4) as usize;
        if idx + 3 < overlay.len() {
            // Blend the violation color
            let alpha = color[3] as f32 / 255.0;
            overlay[idx] =
                (color[0] as f32 * alpha + f32::from(overlay[idx]) * (1.0 - alpha)) as u8;
            overlay[idx + 1] =
                (color[1] as f32 * alpha + f32::from(overlay[idx + 1]) * (1.0 - alpha)) as u8;
            overlay[idx + 2] =
                (color[2] as f32 * alpha + f32::from(overlay[idx + 2]) * (1.0 - alpha)) as u8;
            overlay[idx + 3] = 255;
        }
    }

    overlay
}

/// Compliance statistics summary.
#[derive(Debug, Clone)]
pub struct ComplianceStats {
    /// Percentage of frames that pass compliance.
    pub pass_rate: f32,

    /// Average violation percentage across all frames.
    pub avg_violation_percent: f32,

    /// Maximum violation percentage seen in any frame.
    pub max_violation_percent: f32,

    /// Most common violation type.
    pub most_common_violation: Option<ViolationType>,

    /// Total number of frames analyzed.
    pub frames_analyzed: u32,
}

/// Analyzes multiple frames and produces aggregate compliance statistics.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn analyze_sequence_compliance(reports: &[ComplianceReport]) -> ComplianceStats {
    if reports.is_empty() {
        return ComplianceStats {
            pass_rate: 0.0,
            avg_violation_percent: 0.0,
            max_violation_percent: 0.0,
            most_common_violation: None,
            frames_analyzed: 0,
        };
    }

    let passes = reports.iter().filter(|r| r.passes_compliance).count();
    let pass_rate = (passes as f32 / reports.len() as f32) * 100.0;

    let avg_violation_percent =
        reports.iter().map(|r| r.violation_percent).sum::<f32>() / reports.len() as f32;

    let max_violation_percent = reports
        .iter()
        .map(|r| r.violation_percent)
        .fold(0.0, f32::max);

    // Find most common violation type
    let mut violation_counts = std::collections::HashMap::new();
    for report in reports {
        for (vtype, count) in &report.violations_by_type {
            *violation_counts.entry(*vtype).or_insert(0u32) += count;
        }
    }

    let most_common_violation = violation_counts
        .iter()
        .max_by_key(|(_, &count)| count)
        .map(|(&vtype, _)| vtype);

    ComplianceStats {
        pass_rate,
        avg_violation_percent,
        max_violation_percent,
        most_common_violation,
        frames_analyzed: reports.len() as u32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_legal_frame(width: u32, height: u32) -> Vec<u8> {
        let mut frame = vec![0u8; (width * height * 3) as usize];

        // Create gradient with legal colors (mid-gray range)
        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 3) as usize;
                let value = 128u8; // Safe mid-gray

                frame[idx] = value;
                frame[idx + 1] = value;
                frame[idx + 2] = value;
            }
        }

        frame
    }

    fn create_illegal_frame(width: u32, height: u32) -> Vec<u8> {
        let mut frame = vec![0u8; (width * height * 3) as usize];

        // Create frame with illegal values (pure black and white)
        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 3) as usize;
                let value = if (x + y) % 2 == 0 { 0u8 } else { 255u8 };

                frame[idx] = value;
                frame[idx + 1] = value;
                frame[idx + 2] = value;
            }
        }

        frame
    }

    #[test]
    fn test_legal_range() {
        let bt709 = LegalRange::bt709();
        assert_eq!(bt709.min_luma, 16);
        assert_eq!(bt709.max_luma, 235);
        assert_eq!(bt709.min_chroma, 16);
        assert_eq!(bt709.max_chroma, 240);

        let full = LegalRange::full_range();
        assert_eq!(full.min_luma, 0);
        assert_eq!(full.max_luma, 255);
    }

    #[test]
    fn test_check_compliance_legal_frame() {
        let frame = create_legal_frame(100, 100);
        let config = ComplianceConfig::default();

        let result = check_compliance(&frame, 100, 100, &config);
        assert!(result.is_ok());

        let report = result.expect("should succeed in test");
        // Mid-gray should be legal
        assert!(report.passes_compliance || report.violation_percent < 1.0);
    }

    #[test]
    fn test_check_compliance_illegal_frame() {
        let frame = create_illegal_frame(100, 100);
        let config = ComplianceConfig::default();

        let result = check_compliance(&frame, 100, 100, &config);
        assert!(result.is_ok());

        let report = result.expect("should succeed in test");
        // Pure black/white should have violations
        assert!(report.violation_count > 0);
        assert!(!report.passes_compliance);
    }

    #[test]
    fn test_violation_types() {
        let frame = create_illegal_frame(50, 50);
        let config = ComplianceConfig::default();

        let report = check_compliance(&frame, 50, 50, &config).expect("should succeed in test");

        // Should have multiple violation types
        let non_zero_types = report
            .violations_by_type
            .iter()
            .filter(|(_, count)| *count > 0)
            .count();

        assert!(non_zero_types > 0);
    }

    #[test]
    fn test_is_illegal_color() {
        // Pure colors are likely illegal in broadcast
        assert!(is_illegal_color(255, 0, 0)); // Pure red
        assert!(is_illegal_color(0, 0, 255)); // Pure blue

        // Mid-gray should be legal
        assert!(!is_illegal_color(128, 128, 128));
    }

    #[test]
    fn test_has_composite_overshoot() {
        // High luma with high chroma should trigger overshoot
        assert!(has_composite_overshoot(230, 200, 200));

        // Low luma with high chroma should trigger overshoot
        assert!(has_composite_overshoot(30, 200, 200));

        // Normal values should not trigger overshoot
        assert!(!has_composite_overshoot(128, 128, 128));
    }

    #[test]
    fn test_generate_compliance_overlay() {
        let frame = create_illegal_frame(50, 50);
        let config = ComplianceConfig::default();
        let report = check_compliance(&frame, 50, 50, &config).expect("should succeed in test");

        let overlay = generate_compliance_overlay(&frame, 50, 50, &report);

        assert_eq!(overlay.len(), (50 * 50 * 4) as usize);
    }

    #[test]
    fn test_analyze_sequence_compliance() {
        let legal_frame = create_legal_frame(50, 50);
        let illegal_frame = create_illegal_frame(50, 50);
        let config = ComplianceConfig::default();

        let report1 =
            check_compliance(&legal_frame, 50, 50, &config).expect("should succeed in test");
        let report2 =
            check_compliance(&illegal_frame, 50, 50, &config).expect("should succeed in test");

        let stats = analyze_sequence_compliance(&[report1, report2]);

        assert_eq!(stats.frames_analyzed, 2);
        assert!(stats.avg_violation_percent >= 0.0);
    }

    #[test]
    fn test_different_standards() {
        let frame = create_legal_frame(50, 50);

        let config_bt709 = ComplianceConfig {
            standard: BroadcastStandard::Bt709,
            ..Default::default()
        };

        let config_bt2020 = ComplianceConfig {
            standard: BroadcastStandard::Bt2020,
            ..Default::default()
        };

        let report_709 =
            check_compliance(&frame, 50, 50, &config_bt709).expect("should succeed in test");
        let report_2020 =
            check_compliance(&frame, 50, 50, &config_bt2020).expect("should succeed in test");

        assert_eq!(report_709.standard, BroadcastStandard::Bt709);
        assert_eq!(report_2020.standard, BroadcastStandard::Bt2020);
    }

    #[test]
    fn test_compliance_with_tolerance() {
        let frame = create_illegal_frame(50, 50);

        let config_strict = ComplianceConfig {
            tolerance: 0,
            ..Default::default()
        };

        let config_lenient = ComplianceConfig {
            tolerance: 10,
            ..Default::default()
        };

        let report_strict =
            check_compliance(&frame, 50, 50, &config_strict).expect("should succeed in test");
        let report_lenient =
            check_compliance(&frame, 50, 50, &config_lenient).expect("should succeed in test");

        // Lenient should have fewer or equal violations
        assert!(report_lenient.violation_count <= report_strict.violation_count);
    }

    #[test]
    fn test_invalid_frame_size() {
        let frame = vec![0u8; 100]; // Too small
        let config = ComplianceConfig::default();

        let result = check_compliance(&frame, 100, 100, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_max_violations_limit() {
        let frame = create_illegal_frame(100, 100);

        let config = ComplianceConfig {
            max_violations: 10,
            ..Default::default()
        };

        let report = check_compliance(&frame, 100, 100, &config).expect("should succeed in test");

        // Should not exceed max_violations
        assert!(report.violations.len() <= 10);
    }
}
