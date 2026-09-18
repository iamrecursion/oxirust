#![allow(dead_code)]
//! Loudness standard definitions and conversion utilities.
//!
//! [`LoudnessStandard`] enumerates the most widely used delivery loudness
//! targets.  [`TargetLoudness`] wraps the numeric parameters (LUFS, dBTP,
//! tolerance) for a chosen standard, and [`LoudnessConverter`] translates a
//! measured loudness value from one standard's reference point to another.

// ─────────────────────────────────────────────────────────────────────────────
// LoudnessStandard
// ─────────────────────────────────────────────────────────────────────────────

/// Industry loudness delivery standards.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LoudnessStandard {
    /// EBU R128 — European broadcast: −23 LUFS, −1 dBTP.
    EbuR128,
    /// ATSC A/85 — US broadcast: −24 LKFS, −2 dBTP.
    Atsc,
    /// Netflix — streaming drama: −27 LUFS, −2 dBTP.
    Netflix,
    /// Spotify — streaming music: −14 LUFS, −1 dBTP.
    Spotify,
    /// YouTube — online video: −14 LUFS, −1 dBTP.
    YouTube,
    /// Apple Music / iTunes: −16 LUFS, −1 dBTP.
    AppleMusic,
    /// Amazon Prime Video: −24 LUFS, −2 dBTP.
    AmazonPrime,
    /// BBC iPlayer: −23 LUFS, −1 dBTP (same target as EBU R128).
    BbcIPlayer,
    /// ReplayGain reference level: −18 LUFS, −1 dBTP.
    ReplayGain,
    /// User-defined target.
    Custom {
        /// Target integrated loudness in LUFS.
        target_lufs: f64,
        /// Maximum true peak in dBTP.
        max_peak_dbtp: f64,
        /// Permitted tolerance in LU.
        tolerance_lu: f64,
    },
}

impl LoudnessStandard {
    /// Target integrated loudness in LUFS.
    pub fn target_lufs(self) -> f64 {
        match self {
            Self::EbuR128 | Self::BbcIPlayer => -23.0,
            Self::Atsc | Self::AmazonPrime => -24.0,
            Self::Netflix => -27.0,
            Self::Spotify | Self::YouTube => -14.0,
            Self::AppleMusic => -16.0,
            Self::ReplayGain => -18.0,
            Self::Custom { target_lufs, .. } => target_lufs,
        }
    }

    /// Maximum true-peak level in dBTP.
    pub fn max_true_peak_dbtp(self) -> f64 {
        match self {
            Self::EbuR128
            | Self::BbcIPlayer
            | Self::Spotify
            | Self::YouTube
            | Self::AppleMusic
            | Self::ReplayGain => -1.0,
            Self::Atsc | Self::AmazonPrime | Self::Netflix => -2.0,
            Self::Custom { max_peak_dbtp, .. } => max_peak_dbtp,
        }
    }

    /// Loudness tolerance in LU (±).
    pub fn tolerance_lu(self) -> f64 {
        match self {
            Self::Atsc | Self::AmazonPrime => 2.0,
            Self::Custom { tolerance_lu, .. } => tolerance_lu,
            _ => 1.0,
        }
    }

    /// Canonical display name for the standard.
    pub fn name(self) -> &'static str {
        match self {
            Self::EbuR128 => "EBU R128",
            Self::Atsc => "ATSC A/85",
            Self::Netflix => "Netflix",
            Self::Spotify => "Spotify",
            Self::YouTube => "YouTube",
            Self::AppleMusic => "Apple Music",
            Self::AmazonPrime => "Amazon Prime Video",
            Self::BbcIPlayer => "BBC iPlayer",
            Self::ReplayGain => "ReplayGain",
            Self::Custom { .. } => "Custom",
        }
    }

    /// All non-custom standard variants.
    pub fn all_named() -> &'static [Self] {
        &[
            Self::EbuR128,
            Self::Atsc,
            Self::Netflix,
            Self::Spotify,
            Self::YouTube,
            Self::AppleMusic,
            Self::AmazonPrime,
            Self::BbcIPlayer,
            Self::ReplayGain,
        ]
    }
}

impl std::fmt::Display for LoudnessStandard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TargetLoudness
// ─────────────────────────────────────────────────────────────────────────────

/// Numeric loudness target derived from a [`LoudnessStandard`].
///
/// ```rust
/// use oximedia_normalize::target_loudness::{LoudnessStandard, TargetLoudness};
///
/// let target = TargetLoudness::from_standard(LoudnessStandard::EbuR128);
/// assert_eq!(target.target_lufs, -23.0);
/// assert!(target.is_compliant(-22.5));
/// ```
#[derive(Debug, Clone)]
pub struct TargetLoudness {
    /// The loudness standard this target was derived from.
    pub standard: LoudnessStandard,
    /// Target integrated loudness in LUFS.
    pub target_lufs: f64,
    /// Maximum true peak in dBTP.
    pub max_true_peak_dbtp: f64,
    /// Permitted loudness tolerance in LU.
    pub tolerance_lu: f64,
}

impl TargetLoudness {
    /// Build a [`TargetLoudness`] from a [`LoudnessStandard`].
    pub fn from_standard(standard: LoudnessStandard) -> Self {
        Self {
            target_lufs: standard.target_lufs(),
            max_true_peak_dbtp: standard.max_true_peak_dbtp(),
            tolerance_lu: standard.tolerance_lu(),
            standard,
        }
    }

    /// Build a custom target directly from numeric parameters.
    pub fn custom(target_lufs: f64, max_true_peak_dbtp: f64, tolerance_lu: f64) -> Self {
        Self::from_standard(LoudnessStandard::Custom {
            target_lufs,
            max_peak_dbtp: max_true_peak_dbtp,
            tolerance_lu,
        })
    }

    /// Returns `true` when `measured_lufs` is within the permitted tolerance.
    pub fn is_compliant(&self, measured_lufs: f64) -> bool {
        (measured_lufs - self.target_lufs).abs() <= self.tolerance_lu
    }

    /// Gain in dB required to bring `measured_lufs` to this target.
    pub fn required_gain_db(&self, measured_lufs: f64) -> f64 {
        self.target_lufs - measured_lufs
    }

    /// Upper bound of the compliance window.
    pub fn upper_bound_lufs(&self) -> f64 {
        self.target_lufs + self.tolerance_lu
    }

    /// Lower bound of the compliance window.
    pub fn lower_bound_lufs(&self) -> f64 {
        self.target_lufs - self.tolerance_lu
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LoudnessConverter
// ─────────────────────────────────────────────────────────────────────────────

/// Converts an integrated-loudness measurement between two loudness standards.
///
/// The conversion is purely an offset in LUFS — it tells you what gain to
/// apply so that audio normalised for `from` would pass the `to` standard.
///
/// ```rust
/// use oximedia_normalize::target_loudness::{LoudnessConverter, LoudnessStandard};
///
/// let converter = LoudnessConverter::new(LoudnessStandard::EbuR128, LoudnessStandard::Spotify);
/// // EBU R128 is -23 LUFS; Spotify is -14 LUFS → +9 dB gain needed.
/// assert!((converter.gain_offset_db() - 9.0).abs() < 1e-9);
/// ```
#[derive(Debug, Clone)]
pub struct LoudnessConverter {
    /// Source standard.
    pub from: LoudnessStandard,
    /// Destination standard.
    pub to: LoudnessStandard,
}

impl LoudnessConverter {
    /// Create a converter from `from` to `to`.
    pub fn new(from: LoudnessStandard, to: LoudnessStandard) -> Self {
        Self { from, to }
    }

    /// The fixed gain offset in dB needed to go from `from` to `to`.
    ///
    /// A positive value means the destination is louder than the source.
    pub fn gain_offset_db(&self) -> f64 {
        self.to.target_lufs() - self.from.target_lufs()
    }

    /// Convert a loudness value measured against `from` to the equivalent
    /// level for `to`.
    pub fn convert(&self, lufs: f64) -> f64 {
        lufs + self.gain_offset_db()
    }

    /// Returns `true` when the two standards share the same target LUFS.
    pub fn is_lossless(&self) -> bool {
        (self.from.target_lufs() - self.to.target_lufs()).abs() < 1e-9
    }

    /// Returns `true` when converting from a louder standard to a quieter one
    /// (i.e. gain must be reduced).
    pub fn is_gain_reduction(&self) -> bool {
        self.gain_offset_db() < 0.0
    }

    /// Describe the conversion as a human-readable string.
    pub fn description(&self) -> String {
        let offset = self.gain_offset_db();
        let sign = if offset >= 0.0 { "+" } else { "" };
        format!(
            "{} → {} ({}{:.1} dB)",
            self.from.name(),
            self.to.name(),
            sign,
            offset
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_target_lufs() {
        assert!((LoudnessStandard::EbuR128.target_lufs() - (-23.0)).abs() < 1e-9);
        assert!((LoudnessStandard::Atsc.target_lufs() - (-24.0)).abs() < 1e-9);
        assert!((LoudnessStandard::Netflix.target_lufs() - (-27.0)).abs() < 1e-9);
        assert!((LoudnessStandard::Spotify.target_lufs() - (-14.0)).abs() < 1e-9);
    }

    #[test]
    fn test_standard_max_true_peak() {
        assert!((LoudnessStandard::EbuR128.max_true_peak_dbtp() - (-1.0)).abs() < 1e-9);
        assert!((LoudnessStandard::Atsc.max_true_peak_dbtp() - (-2.0)).abs() < 1e-9);
    }

    #[test]
    fn test_standard_tolerance() {
        assert!((LoudnessStandard::EbuR128.tolerance_lu() - 1.0).abs() < 1e-9);
        assert!((LoudnessStandard::Atsc.tolerance_lu() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_standard_name() {
        assert_eq!(LoudnessStandard::EbuR128.name(), "EBU R128");
        assert_eq!(LoudnessStandard::Netflix.name(), "Netflix");
    }

    #[test]
    fn test_standard_display() {
        assert_eq!(LoudnessStandard::Spotify.to_string(), "Spotify");
    }

    #[test]
    fn test_all_named_count() {
        assert_eq!(LoudnessStandard::all_named().len(), 9);
    }

    #[test]
    fn test_custom_standard() {
        let custom = LoudnessStandard::Custom {
            target_lufs: -20.0,
            max_peak_dbtp: -1.5,
            tolerance_lu: 0.5,
        };
        assert!((custom.target_lufs() - (-20.0)).abs() < 1e-9);
        assert!((custom.tolerance_lu() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_target_from_standard() {
        let target = TargetLoudness::from_standard(LoudnessStandard::EbuR128);
        assert!((target.target_lufs - (-23.0)).abs() < 1e-9);
        assert!((target.max_true_peak_dbtp - (-1.0)).abs() < 1e-9);
    }

    #[test]
    fn test_target_is_compliant() {
        let target = TargetLoudness::from_standard(LoudnessStandard::EbuR128);
        assert!(target.is_compliant(-23.0));
        assert!(target.is_compliant(-22.5));
        assert!(target.is_compliant(-23.9));
        assert!(!target.is_compliant(-20.0));
        assert!(!target.is_compliant(-25.5));
    }

    #[test]
    fn test_target_required_gain() {
        let target = TargetLoudness::from_standard(LoudnessStandard::EbuR128);
        assert!((target.required_gain_db(-26.0) - 3.0).abs() < 1e-9);
        assert!((target.required_gain_db(-20.0) - (-3.0)).abs() < 1e-9);
    }

    #[test]
    fn test_target_bounds() {
        let target = TargetLoudness::from_standard(LoudnessStandard::EbuR128);
        assert!((target.upper_bound_lufs() - (-22.0)).abs() < 1e-9);
        assert!((target.lower_bound_lufs() - (-24.0)).abs() < 1e-9);
    }

    #[test]
    fn test_target_custom() {
        let target = TargetLoudness::custom(-20.0, -1.5, 0.5);
        assert!((target.target_lufs - (-20.0)).abs() < 1e-9);
    }

    #[test]
    fn test_converter_gain_offset_ebu_to_spotify() {
        let converter =
            LoudnessConverter::new(LoudnessStandard::EbuR128, LoudnessStandard::Spotify);
        // -14 - (-23) = +9
        assert!((converter.gain_offset_db() - 9.0).abs() < 1e-9);
    }

    #[test]
    fn test_converter_is_lossless_for_same() {
        let converter =
            LoudnessConverter::new(LoudnessStandard::EbuR128, LoudnessStandard::BbcIPlayer);
        assert!(converter.is_lossless());
    }

    #[test]
    fn test_converter_is_not_lossless() {
        let converter =
            LoudnessConverter::new(LoudnessStandard::EbuR128, LoudnessStandard::Netflix);
        assert!(!converter.is_lossless());
    }

    #[test]
    fn test_converter_is_gain_reduction() {
        // Spotify (-14) → Netflix (-27): must reduce gain.
        let converter =
            LoudnessConverter::new(LoudnessStandard::Spotify, LoudnessStandard::Netflix);
        assert!(converter.is_gain_reduction());
    }

    #[test]
    fn test_converter_convert_value() {
        let converter = LoudnessConverter::new(LoudnessStandard::EbuR128, LoudnessStandard::Atsc);
        // EbuR128 = -23, Atsc = -24 → offset -1.
        let result = converter.convert(-23.0);
        assert!((result - (-24.0)).abs() < 1e-9);
    }

    #[test]
    fn test_converter_description_contains_arrow() {
        let converter =
            LoudnessConverter::new(LoudnessStandard::EbuR128, LoudnessStandard::Spotify);
        let desc = converter.description();
        assert!(desc.contains("→"));
        assert!(desc.contains("EBU R128"));
        assert!(desc.contains("Spotify"));
    }
}
