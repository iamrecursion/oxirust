#![allow(dead_code)]
//! Codec profile and level definitions.
//!
//! Defines the profile/level/tier system used by modern video codecs to
//! describe capability requirements. Provides constraint checking so that
//! an encoder can verify the output conforms to a chosen profile/level pair.

use std::fmt;

/// Video codec family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CodecFamily {
    /// AV1 codec.
    Av1,
    /// H.264 / AVC codec.
    H264,
    /// H.265 / HEVC codec.
    H265,
    /// VP9 codec.
    Vp9,
}

impl fmt::Display for CodecFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Av1 => write!(f, "AV1"),
            Self::H264 => write!(f, "H.264"),
            Self::H265 => write!(f, "H.265"),
            Self::Vp9 => write!(f, "VP9"),
        }
    }
}

/// Generic codec profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Profile {
    /// Main profile (8-bit 4:2:0).
    Main,
    /// Main 10 profile (10-bit 4:2:0).
    Main10,
    /// High profile (8-bit 4:2:0 with advanced coding tools).
    High,
    /// Professional profile (4:2:2 / 4:4:4).
    Professional,
    /// Baseline profile (constrained feature set).
    Baseline,
}

impl fmt::Display for Profile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Main => write!(f, "Main"),
            Self::Main10 => write!(f, "Main10"),
            Self::High => write!(f, "High"),
            Self::Professional => write!(f, "Professional"),
            Self::Baseline => write!(f, "Baseline"),
        }
    }
}

/// Generic codec level as a numeric identifier.
///
/// The numeric value follows codec conventions (e.g., H.264 level 5.1 = 51).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Level(pub u16);

impl Level {
    /// Level 3.0.
    pub const L3_0: Self = Self(30);
    /// Level 3.1.
    pub const L3_1: Self = Self(31);
    /// Level 4.0.
    pub const L4_0: Self = Self(40);
    /// Level 4.1.
    pub const L4_1: Self = Self(41);
    /// Level 5.0.
    pub const L5_0: Self = Self(50);
    /// Level 5.1.
    pub const L5_1: Self = Self(51);
    /// Level 5.2.
    pub const L5_2: Self = Self(52);
    /// Level 6.0.
    pub const L6_0: Self = Self(60);
    /// Level 6.1.
    pub const L6_1: Self = Self(61);
}

impl fmt::Display for Level {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.0 / 10, self.0 % 10)
    }
}

/// Codec tier (main or high).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Tier {
    /// Main tier (lower bitrate constraints).
    Main,
    /// High tier (higher bitrate constraints).
    High,
}

impl Default for Tier {
    fn default() -> Self {
        Self::Main
    }
}

/// Constraints imposed by a profile/level combination.
#[derive(Clone, Debug)]
pub struct LevelConstraints {
    /// Maximum picture width in pixels.
    pub max_width: u32,
    /// Maximum picture height in pixels.
    pub max_height: u32,
    /// Maximum luma sample rate (samples per second).
    pub max_sample_rate: u64,
    /// Maximum bitrate in bits per second for main tier.
    pub max_bitrate_main: u64,
    /// Maximum bitrate in bits per second for high tier.
    pub max_bitrate_high: u64,
    /// Maximum number of reference frames.
    pub max_ref_frames: u8,
    /// Maximum number of tile columns.
    pub max_tile_cols: u8,
    /// Maximum number of tile rows.
    pub max_tile_rows: u8,
}

impl LevelConstraints {
    /// Maximum bitrate for the given tier.
    pub fn max_bitrate(&self, tier: Tier) -> u64 {
        match tier {
            Tier::Main => self.max_bitrate_main,
            Tier::High => self.max_bitrate_high,
        }
    }

    /// Maximum total pixel count (width * height).
    pub fn max_pixels(&self) -> u64 {
        u64::from(self.max_width) * u64::from(self.max_height)
    }
}

/// Get level constraints for a given codec and level.
pub fn get_constraints(codec: CodecFamily, level: Level) -> LevelConstraints {
    match codec {
        CodecFamily::Av1 => av1_constraints(level),
        CodecFamily::H264 => h264_constraints(level),
        CodecFamily::H265 => h265_constraints(level),
        CodecFamily::Vp9 => vp9_constraints(level),
    }
}

fn av1_constraints(level: Level) -> LevelConstraints {
    match level.0 {
        50 => LevelConstraints {
            max_width: 4096,
            max_height: 2304,
            max_sample_rate: 66_846_720,
            max_bitrate_main: 30_000_000,
            max_bitrate_high: 60_000_000,
            max_ref_frames: 7,
            max_tile_cols: 8,
            max_tile_rows: 8,
        },
        51 => LevelConstraints {
            max_width: 4096,
            max_height: 2304,
            max_sample_rate: 133_693_440,
            max_bitrate_main: 40_000_000,
            max_bitrate_high: 100_000_000,
            max_ref_frames: 7,
            max_tile_cols: 8,
            max_tile_rows: 8,
        },
        60 => LevelConstraints {
            max_width: 8192,
            max_height: 4352,
            max_sample_rate: 267_386_880,
            max_bitrate_main: 60_000_000,
            max_bitrate_high: 180_000_000,
            max_ref_frames: 7,
            max_tile_cols: 16,
            max_tile_rows: 16,
        },
        _ => default_constraints(),
    }
}

fn h264_constraints(level: Level) -> LevelConstraints {
    match level.0 {
        40 => LevelConstraints {
            max_width: 2048,
            max_height: 1024,
            max_sample_rate: 62_914_560,
            max_bitrate_main: 20_000_000,
            max_bitrate_high: 25_000_000,
            max_ref_frames: 4,
            max_tile_cols: 1,
            max_tile_rows: 1,
        },
        51 => LevelConstraints {
            max_width: 4096,
            max_height: 2304,
            max_sample_rate: 251_658_240,
            max_bitrate_main: 135_000_000,
            max_bitrate_high: 240_000_000,
            max_ref_frames: 16,
            max_tile_cols: 1,
            max_tile_rows: 1,
        },
        _ => default_constraints(),
    }
}

fn h265_constraints(level: Level) -> LevelConstraints {
    match level.0 {
        40 => LevelConstraints {
            max_width: 2048,
            max_height: 1080,
            max_sample_rate: 62_914_560,
            max_bitrate_main: 12_000_000,
            max_bitrate_high: 30_000_000,
            max_ref_frames: 6,
            max_tile_cols: 5,
            max_tile_rows: 5,
        },
        51 => LevelConstraints {
            max_width: 4096,
            max_height: 2160,
            max_sample_rate: 267_386_880,
            max_bitrate_main: 40_000_000,
            max_bitrate_high: 160_000_000,
            max_ref_frames: 6,
            max_tile_cols: 10,
            max_tile_rows: 10,
        },
        _ => default_constraints(),
    }
}

fn vp9_constraints(level: Level) -> LevelConstraints {
    match level.0 {
        40 => LevelConstraints {
            max_width: 2048,
            max_height: 1080,
            max_sample_rate: 62_914_560,
            max_bitrate_main: 18_000_000,
            max_bitrate_high: 36_000_000,
            max_ref_frames: 3,
            max_tile_cols: 4,
            max_tile_rows: 4,
        },
        51 => LevelConstraints {
            max_width: 4096,
            max_height: 2160,
            max_sample_rate: 267_386_880,
            max_bitrate_main: 36_000_000,
            max_bitrate_high: 120_000_000,
            max_ref_frames: 3,
            max_tile_cols: 8,
            max_tile_rows: 8,
        },
        _ => default_constraints(),
    }
}

fn default_constraints() -> LevelConstraints {
    LevelConstraints {
        max_width: 1920,
        max_height: 1080,
        max_sample_rate: 31_457_280,
        max_bitrate_main: 10_000_000,
        max_bitrate_high: 20_000_000,
        max_ref_frames: 4,
        max_tile_cols: 1,
        max_tile_rows: 1,
    }
}

/// Result of a constraint compliance check.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComplianceResult {
    /// Whether the configuration passes all checks.
    pub passed: bool,
    /// List of violated constraints.
    pub violations: Vec<String>,
}

impl ComplianceResult {
    /// Create a passing result with no violations.
    pub fn pass() -> Self {
        Self {
            passed: true,
            violations: Vec::new(),
        }
    }

    /// Create a failing result with the given violations.
    pub fn fail(violations: Vec<String>) -> Self {
        Self {
            passed: false,
            violations,
        }
    }
}

/// Check whether a given encoding configuration fits within a level's constraints.
#[allow(clippy::cast_precision_loss)]
pub fn check_compliance(
    codec: CodecFamily,
    level: Level,
    tier: Tier,
    width: u32,
    height: u32,
    framerate: f64,
    bitrate: u64,
) -> ComplianceResult {
    let constraints = get_constraints(codec, level);
    let mut violations = Vec::new();

    if width > constraints.max_width {
        violations.push(format!(
            "Width {width} exceeds max {max}",
            max = constraints.max_width
        ));
    }
    if height > constraints.max_height {
        violations.push(format!(
            "Height {height} exceeds max {max}",
            max = constraints.max_height
        ));
    }

    let sample_rate = (u64::from(width) * u64::from(height)) as f64 * framerate;
    if sample_rate > constraints.max_sample_rate as f64 {
        violations.push(format!(
            "Sample rate {sample_rate:.0} exceeds max {}",
            constraints.max_sample_rate
        ));
    }

    let max_br = constraints.max_bitrate(tier);
    if bitrate > max_br {
        violations.push(format!(
            "Bitrate {bitrate} exceeds max {max_br} for tier {tier:?}"
        ));
    }

    if violations.is_empty() {
        ComplianceResult::pass()
    } else {
        ComplianceResult::fail(violations)
    }
}

/// Find the minimum level that satisfies the given parameters.
#[allow(clippy::cast_precision_loss)]
pub fn find_minimum_level(
    codec: CodecFamily,
    tier: Tier,
    width: u32,
    height: u32,
    framerate: f64,
    bitrate: u64,
) -> Option<Level> {
    let candidates = [
        Level::L3_0,
        Level::L3_1,
        Level::L4_0,
        Level::L4_1,
        Level::L5_0,
        Level::L5_1,
        Level::L5_2,
        Level::L6_0,
        Level::L6_1,
    ];
    for lvl in &candidates {
        let result = check_compliance(codec, *lvl, tier, width, height, framerate, bitrate);
        if result.passed {
            return Some(*lvl);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codec_family_display() {
        assert_eq!(CodecFamily::Av1.to_string(), "AV1");
        assert_eq!(CodecFamily::H264.to_string(), "H.264");
        assert_eq!(CodecFamily::H265.to_string(), "H.265");
        assert_eq!(CodecFamily::Vp9.to_string(), "VP9");
    }

    #[test]
    fn test_profile_display() {
        assert_eq!(Profile::Main.to_string(), "Main");
        assert_eq!(Profile::Main10.to_string(), "Main10");
        assert_eq!(Profile::Baseline.to_string(), "Baseline");
    }

    #[test]
    fn test_level_display() {
        assert_eq!(Level::L5_1.to_string(), "5.1");
        assert_eq!(Level::L4_0.to_string(), "4.0");
        assert_eq!(Level::L6_1.to_string(), "6.1");
    }

    #[test]
    fn test_level_ordering() {
        assert!(Level::L3_0 < Level::L4_0);
        assert!(Level::L5_0 < Level::L5_1);
        assert!(Level::L5_1 < Level::L6_0);
    }

    #[test]
    fn test_tier_default() {
        assert_eq!(Tier::default(), Tier::Main);
    }

    #[test]
    fn test_av1_level_50_constraints() {
        let c = get_constraints(CodecFamily::Av1, Level::L5_0);
        assert_eq!(c.max_width, 4096);
        assert_eq!(c.max_height, 2304);
        assert_eq!(c.max_ref_frames, 7);
    }

    #[test]
    fn test_max_bitrate_tier() {
        let c = get_constraints(CodecFamily::Av1, Level::L5_0);
        assert_eq!(c.max_bitrate(Tier::Main), 30_000_000);
        assert_eq!(c.max_bitrate(Tier::High), 60_000_000);
    }

    #[test]
    fn test_max_pixels() {
        let c = get_constraints(CodecFamily::Av1, Level::L5_0);
        assert_eq!(c.max_pixels(), 4096 * 2304);
    }

    #[test]
    fn test_compliance_pass() {
        let result = check_compliance(
            CodecFamily::Av1,
            Level::L5_1,
            Tier::Main,
            1920,
            1080,
            60.0,
            10_000_000,
        );
        assert!(result.passed);
        assert!(result.violations.is_empty());
    }

    #[test]
    fn test_compliance_fail_width() {
        let result = check_compliance(
            CodecFamily::Av1,
            Level::L5_0,
            Tier::Main,
            8192,
            4320,
            30.0,
            10_000_000,
        );
        assert!(!result.passed);
        assert!(!result.violations.is_empty());
    }

    #[test]
    fn test_compliance_fail_bitrate() {
        let result = check_compliance(
            CodecFamily::Av1,
            Level::L5_0,
            Tier::Main,
            1920,
            1080,
            30.0,
            999_000_000,
        );
        assert!(!result.passed);
        assert!(result.violations.iter().any(|v| v.contains("Bitrate")));
    }

    #[test]
    fn test_find_minimum_level() {
        let lvl = find_minimum_level(CodecFamily::Av1, Tier::Main, 1920, 1080, 30.0, 8_000_000);
        assert!(lvl.is_some());
    }

    #[test]
    fn test_find_minimum_level_none() {
        // Absurdly high bitrate that no level supports
        let lvl = find_minimum_level(
            CodecFamily::Av1,
            Tier::Main,
            8192,
            4320,
            120.0,
            999_999_999_999,
        );
        assert!(lvl.is_none());
    }

    #[test]
    fn test_compliance_result_constructors() {
        let pass = ComplianceResult::pass();
        assert!(pass.passed);
        assert!(pass.violations.is_empty());

        let fail = ComplianceResult::fail(vec!["bad".to_string()]);
        assert!(!fail.passed);
        assert_eq!(fail.violations.len(), 1);
    }
}
