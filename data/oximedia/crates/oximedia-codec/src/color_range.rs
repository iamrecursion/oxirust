#![allow(dead_code)]
//! Color range and level mapping for codec output.
//!
//! Handles the distinction between limited/full range color representations
//! used by video codecs (BT.601/709/2020), and provides conversions between them.
//!
//! # Overview
//!
//! Video codecs typically encode luma in the range \[16, 235\] (limited) while
//! full-range uses \[0, 255\]. This module provides utilities for converting
//! between these representations, verifying compliance, and clamping values.

/// Color range type for video signals.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ColorRange {
    /// Limited range (BT.601/709): Y \[16..235\], UV \[16..240\].
    Limited,
    /// Full range: Y \[0..255\], UV \[0..255\].
    Full,
}

impl Default for ColorRange {
    fn default() -> Self {
        Self::Limited
    }
}

impl ColorRange {
    /// Returns true if this is limited range.
    pub fn is_limited(&self) -> bool {
        *self == Self::Limited
    }

    /// Returns true if this is full range.
    pub fn is_full(&self) -> bool {
        *self == Self::Full
    }
}

/// Bit depth for color levels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BitDepth {
    /// 8-bit depth (0..255).
    Eight,
    /// 10-bit depth (0..1023).
    Ten,
    /// 12-bit depth (0..4095).
    Twelve,
}

impl BitDepth {
    /// Maximum value for this bit depth.
    #[allow(clippy::cast_precision_loss)]
    pub fn max_value(&self) -> u16 {
        match self {
            Self::Eight => 255,
            Self::Ten => 1023,
            Self::Twelve => 4095,
        }
    }

    /// Number of bits.
    pub fn bits(&self) -> u8 {
        match self {
            Self::Eight => 8,
            Self::Ten => 10,
            Self::Twelve => 12,
        }
    }
}

impl Default for BitDepth {
    fn default() -> Self {
        Self::Eight
    }
}

/// Level range for a given color range and bit depth.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LevelRange {
    /// Minimum luma value.
    pub luma_min: u16,
    /// Maximum luma value.
    pub luma_max: u16,
    /// Minimum chroma value.
    pub chroma_min: u16,
    /// Maximum chroma value.
    pub chroma_max: u16,
}

impl LevelRange {
    /// Create a level range for the given color range and bit depth.
    #[allow(clippy::cast_precision_loss)]
    pub fn new(range: ColorRange, depth: BitDepth) -> Self {
        let shift = depth.bits() - 8;
        match range {
            ColorRange::Limited => Self {
                luma_min: 16 << shift,
                luma_max: 235 << shift,
                chroma_min: 16 << shift,
                chroma_max: 240 << shift,
            },
            ColorRange::Full => Self {
                luma_min: 0,
                luma_max: depth.max_value(),
                chroma_min: 0,
                chroma_max: depth.max_value(),
            },
        }
    }

    /// Luma span (max - min).
    pub fn luma_span(&self) -> u16 {
        self.luma_max - self.luma_min
    }

    /// Chroma span (max - min).
    pub fn chroma_span(&self) -> u16 {
        self.chroma_max - self.chroma_min
    }
}

/// Clamp a luma value to the given level range.
pub fn clamp_luma(value: u16, levels: &LevelRange) -> u16 {
    value.clamp(levels.luma_min, levels.luma_max)
}

/// Clamp a chroma value to the given level range.
pub fn clamp_chroma(value: u16, levels: &LevelRange) -> u16 {
    value.clamp(levels.chroma_min, levels.chroma_max)
}

/// Convert a luma value from limited range to full range.
#[allow(clippy::cast_precision_loss)]
pub fn limited_to_full_luma(value: u16, depth: BitDepth) -> u16 {
    let limited = LevelRange::new(ColorRange::Limited, depth);
    let max = depth.max_value();
    if value <= limited.luma_min {
        return 0;
    }
    if value >= limited.luma_max {
        return max;
    }
    let span = limited.luma_span() as f64;
    let scaled = (value - limited.luma_min) as f64 / span * max as f64;
    (scaled.round() as u16).min(max)
}

/// Convert a luma value from full range to limited range.
#[allow(clippy::cast_precision_loss)]
pub fn full_to_limited_luma(value: u16, depth: BitDepth) -> u16 {
    let limited = LevelRange::new(ColorRange::Limited, depth);
    let max = depth.max_value();
    let span = limited.luma_span() as f64;
    let scaled = value as f64 / max as f64 * span + limited.luma_min as f64;
    (scaled.round() as u16).clamp(limited.luma_min, limited.luma_max)
}

/// Convert a chroma value from limited range to full range.
#[allow(clippy::cast_precision_loss)]
pub fn limited_to_full_chroma(value: u16, depth: BitDepth) -> u16 {
    let limited = LevelRange::new(ColorRange::Limited, depth);
    let max = depth.max_value();
    if value <= limited.chroma_min {
        return 0;
    }
    if value >= limited.chroma_max {
        return max;
    }
    let span = limited.chroma_span() as f64;
    let scaled = (value - limited.chroma_min) as f64 / span * max as f64;
    (scaled.round() as u16).min(max)
}

/// Convert a chroma value from full range to limited range.
#[allow(clippy::cast_precision_loss)]
pub fn full_to_limited_chroma(value: u16, depth: BitDepth) -> u16 {
    let limited = LevelRange::new(ColorRange::Limited, depth);
    let max = depth.max_value();
    let span = limited.chroma_span() as f64;
    let scaled = value as f64 / max as f64 * span + limited.chroma_min as f64;
    (scaled.round() as u16).clamp(limited.chroma_min, limited.chroma_max)
}

/// Result of a compliance check on a buffer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComplianceReport {
    /// Number of out-of-range luma samples.
    pub luma_violations: usize,
    /// Number of out-of-range chroma samples.
    pub chroma_violations: usize,
    /// Total samples checked.
    pub total_samples: usize,
}

impl ComplianceReport {
    /// Returns true if all samples are within the valid range.
    pub fn is_compliant(&self) -> bool {
        self.luma_violations == 0 && self.chroma_violations == 0
    }

    /// Violation ratio as a fraction of total samples.
    #[allow(clippy::cast_precision_loss)]
    pub fn violation_ratio(&self) -> f64 {
        if self.total_samples == 0 {
            return 0.0;
        }
        (self.luma_violations + self.chroma_violations) as f64 / self.total_samples as f64
    }
}

/// Check luma buffer compliance against a level range.
pub fn check_luma_compliance(samples: &[u16], levels: &LevelRange) -> usize {
    samples
        .iter()
        .filter(|&&v| v < levels.luma_min || v > levels.luma_max)
        .count()
}

/// Check chroma buffer compliance against a level range.
pub fn check_chroma_compliance(samples: &[u16], levels: &LevelRange) -> usize {
    samples
        .iter()
        .filter(|&&v| v < levels.chroma_min || v > levels.chroma_max)
        .count()
}

/// Convert an entire luma buffer from one range to another.
#[allow(clippy::cast_precision_loss)]
pub fn convert_luma_buffer(
    src: &[u16],
    src_range: ColorRange,
    dst_range: ColorRange,
    depth: BitDepth,
) -> Vec<u16> {
    if src_range == dst_range {
        return src.to_vec();
    }
    match (src_range, dst_range) {
        (ColorRange::Limited, ColorRange::Full) => src
            .iter()
            .map(|&v| limited_to_full_luma(v, depth))
            .collect(),
        (ColorRange::Full, ColorRange::Limited) => src
            .iter()
            .map(|&v| full_to_limited_luma(v, depth))
            .collect(),
        _ => src.to_vec(),
    }
}

/// Convert an entire chroma buffer from one range to another.
#[allow(clippy::cast_precision_loss)]
pub fn convert_chroma_buffer(
    src: &[u16],
    src_range: ColorRange,
    dst_range: ColorRange,
    depth: BitDepth,
) -> Vec<u16> {
    if src_range == dst_range {
        return src.to_vec();
    }
    match (src_range, dst_range) {
        (ColorRange::Limited, ColorRange::Full) => src
            .iter()
            .map(|&v| limited_to_full_chroma(v, depth))
            .collect(),
        (ColorRange::Full, ColorRange::Limited) => src
            .iter()
            .map(|&v| full_to_limited_chroma(v, depth))
            .collect(),
        _ => src.to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_range_default() {
        let range = ColorRange::default();
        assert_eq!(range, ColorRange::Limited);
    }

    #[test]
    fn test_color_range_predicates() {
        assert!(ColorRange::Limited.is_limited());
        assert!(!ColorRange::Limited.is_full());
        assert!(ColorRange::Full.is_full());
        assert!(!ColorRange::Full.is_limited());
    }

    #[test]
    fn test_bit_depth_max_value() {
        assert_eq!(BitDepth::Eight.max_value(), 255);
        assert_eq!(BitDepth::Ten.max_value(), 1023);
        assert_eq!(BitDepth::Twelve.max_value(), 4095);
    }

    #[test]
    fn test_bit_depth_bits() {
        assert_eq!(BitDepth::Eight.bits(), 8);
        assert_eq!(BitDepth::Ten.bits(), 10);
        assert_eq!(BitDepth::Twelve.bits(), 12);
    }

    #[test]
    fn test_level_range_limited_8bit() {
        let levels = LevelRange::new(ColorRange::Limited, BitDepth::Eight);
        assert_eq!(levels.luma_min, 16);
        assert_eq!(levels.luma_max, 235);
        assert_eq!(levels.chroma_min, 16);
        assert_eq!(levels.chroma_max, 240);
    }

    #[test]
    fn test_level_range_full_8bit() {
        let levels = LevelRange::new(ColorRange::Full, BitDepth::Eight);
        assert_eq!(levels.luma_min, 0);
        assert_eq!(levels.luma_max, 255);
        assert_eq!(levels.chroma_min, 0);
        assert_eq!(levels.chroma_max, 255);
    }

    #[test]
    fn test_level_range_limited_10bit() {
        let levels = LevelRange::new(ColorRange::Limited, BitDepth::Ten);
        assert_eq!(levels.luma_min, 64);
        assert_eq!(levels.luma_max, 940);
        assert_eq!(levels.chroma_min, 64);
        assert_eq!(levels.chroma_max, 960);
    }

    #[test]
    fn test_clamp_luma() {
        let levels = LevelRange::new(ColorRange::Limited, BitDepth::Eight);
        assert_eq!(clamp_luma(0, &levels), 16);
        assert_eq!(clamp_luma(128, &levels), 128);
        assert_eq!(clamp_luma(255, &levels), 235);
    }

    #[test]
    fn test_limited_to_full_luma_8bit() {
        let depth = BitDepth::Eight;
        assert_eq!(limited_to_full_luma(16, depth), 0);
        assert_eq!(limited_to_full_luma(235, depth), 255);
        // Mid-range should map roughly to mid-range
        let mid = limited_to_full_luma(126, depth);
        assert!(mid > 100 && mid < 160);
    }

    #[test]
    fn test_full_to_limited_luma_8bit() {
        let depth = BitDepth::Eight;
        assert_eq!(full_to_limited_luma(0, depth), 16);
        assert_eq!(full_to_limited_luma(255, depth), 235);
    }

    #[test]
    fn test_roundtrip_luma() {
        let depth = BitDepth::Eight;
        for v in (16..=235).step_by(10) {
            let full = limited_to_full_luma(v, depth);
            let back = full_to_limited_luma(full, depth);
            assert!(
                (back as i32 - v as i32).unsigned_abs() <= 1,
                "roundtrip failed for {v}"
            );
        }
    }

    #[test]
    fn test_compliance_report() {
        let report = ComplianceReport {
            luma_violations: 0,
            chroma_violations: 0,
            total_samples: 100,
        };
        assert!(report.is_compliant());
        assert!((report.violation_ratio() - 0.0).abs() < f64::EPSILON);

        let bad = ComplianceReport {
            luma_violations: 5,
            chroma_violations: 3,
            total_samples: 100,
        };
        assert!(!bad.is_compliant());
        assert!((bad.violation_ratio() - 0.08).abs() < f64::EPSILON);
    }

    #[test]
    fn test_check_luma_compliance() {
        let levels = LevelRange::new(ColorRange::Limited, BitDepth::Eight);
        let samples = vec![0, 16, 128, 235, 255];
        let violations = check_luma_compliance(&samples, &levels);
        assert_eq!(violations, 2); // 0 and 255
    }

    #[test]
    fn test_convert_luma_buffer_same_range() {
        let buf = vec![16, 128, 235];
        let result = convert_luma_buffer(
            &buf,
            ColorRange::Limited,
            ColorRange::Limited,
            BitDepth::Eight,
        );
        assert_eq!(result, buf);
    }

    #[test]
    fn test_convert_chroma_buffer() {
        let buf = vec![16, 128, 240];
        let result =
            convert_chroma_buffer(&buf, ColorRange::Limited, ColorRange::Full, BitDepth::Eight);
        assert_eq!(result[0], 0);
        assert_eq!(result[2], 255);
    }
}
