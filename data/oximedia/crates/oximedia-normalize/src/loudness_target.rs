//! Loudness target definitions and compliance checking.
//!
//! Provides `LoudnessTarget`, `LoudnessTargetConfig`, and `LoudnessTargetReport`
//! for specifying delivery loudness targets and validating measurements against them.

#![allow(dead_code)]

/// Loudness delivery target type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LoudnessTarget {
    /// EBU R128 broadcast standard: -23 LUFS integrated.
    EbuR128Broadcast,
    /// Streaming platform standard (e.g. Spotify/YouTube): -14 LUFS integrated.
    StreamingPlatform,
    /// Film mix standard: -27 LUFS integrated.
    FilmMix,
    /// Custom target with explicit LKFS value.
    Custom(f64),
}

impl LoudnessTarget {
    /// Return the integrated loudness target in LKFS / LUFS.
    #[allow(clippy::cast_precision_loss)]
    pub fn target_lkfs(self) -> f64 {
        match self {
            Self::EbuR128Broadcast => -23.0,
            Self::StreamingPlatform => -14.0,
            Self::FilmMix => -27.0,
            Self::Custom(v) => v,
        }
    }

    /// Return the maximum true peak ceiling in dBTP.
    pub fn max_true_peak_dbtp(self) -> f64 {
        match self {
            Self::EbuR128Broadcast => -1.0,
            Self::StreamingPlatform => -1.0,
            Self::FilmMix => -2.0,
            Self::Custom(_) => -1.0,
        }
    }

    /// Return a human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            Self::EbuR128Broadcast => "EBU R128 Broadcast",
            Self::StreamingPlatform => "Streaming Platform",
            Self::FilmMix => "Film Mix",
            Self::Custom(_) => "Custom",
        }
    }
}

/// Configuration for a loudness compliance check.
#[derive(Debug, Clone)]
pub struct LoudnessTargetConfig {
    /// The loudness target.
    pub target: LoudnessTarget,
    /// Allowed tolerance below target in LU (positive value).
    pub tolerance_below_lu: f64,
    /// Allowed tolerance above target in LU (positive value).
    pub tolerance_above_lu: f64,
    /// Whether to enforce true-peak ceiling.
    pub enforce_true_peak: bool,
}

impl LoudnessTargetConfig {
    /// Create a new config with default tolerances for the given target.
    pub fn new(target: LoudnessTarget) -> Self {
        let (below, above) = match target {
            LoudnessTarget::EbuR128Broadcast => (1.0, 1.0),
            LoudnessTarget::StreamingPlatform => (2.0, 1.0),
            LoudnessTarget::FilmMix => (2.0, 2.0),
            LoudnessTarget::Custom(_) => (1.0, 1.0),
        };
        Self {
            target,
            tolerance_below_lu: below,
            tolerance_above_lu: above,
            enforce_true_peak: true,
        }
    }

    /// Create a strict config with ±0.5 LU tolerance.
    pub fn strict(target: LoudnessTarget) -> Self {
        Self {
            target,
            tolerance_below_lu: 0.5,
            tolerance_above_lu: 0.5,
            enforce_true_peak: true,
        }
    }

    /// Return the allowed loudness range as `(min_lkfs, max_lkfs)`.
    pub fn allowed_range(&self) -> (f64, f64) {
        let t = self.target.target_lkfs();
        (t - self.tolerance_below_lu, t + self.tolerance_above_lu)
    }

    /// Return the target bitrate in kbps for a rough proxy of encode budget.
    /// This is a heuristic mapping – not an official standard value.
    pub fn bitrate_kbps(&self) -> u32 {
        match self.target {
            LoudnessTarget::EbuR128Broadcast => 320,
            LoudnessTarget::StreamingPlatform => 256,
            LoudnessTarget::FilmMix => 448,
            LoudnessTarget::Custom(_) => 256,
        }
    }

    /// Check whether a measured loudness value is within the configured tolerance.
    pub fn tolerance_ok(&self, measured_lkfs: f64) -> bool {
        let (min, max) = self.allowed_range();
        measured_lkfs >= min && measured_lkfs <= max
    }
}

/// Report produced after comparing a measurement against a `LoudnessTargetConfig`.
#[derive(Debug, Clone)]
pub struct LoudnessTargetReport {
    /// Configured target.
    pub config: LoudnessTargetConfig,
    /// Measured integrated loudness in LKFS.
    pub measured_lkfs: f64,
    /// Measured true peak in dBTP (if available).
    pub measured_true_peak_dbtp: Option<f64>,
    /// Deviation from target (measured − target).
    pub deviation_lu: f64,
    /// Whether the integrated loudness passes the tolerance check.
    pub integrated_pass: bool,
    /// Whether the true-peak check passes (or was not enforced).
    pub true_peak_pass: bool,
}

impl LoudnessTargetReport {
    /// Build a report from a measurement.
    pub fn new(
        config: LoudnessTargetConfig,
        measured_lkfs: f64,
        measured_true_peak_dbtp: Option<f64>,
    ) -> Self {
        let target_lkfs = config.target.target_lkfs();
        let deviation_lu = measured_lkfs - target_lkfs;
        let integrated_pass = config.tolerance_ok(measured_lkfs);
        let true_peak_pass = if config.enforce_true_peak {
            measured_true_peak_dbtp.map_or(true, |tp| tp <= config.target.max_true_peak_dbtp())
        } else {
            true
        };
        Self {
            config,
            measured_lkfs,
            measured_true_peak_dbtp,
            deviation_lu,
            integrated_pass,
            true_peak_pass,
        }
    }

    /// Return `true` when both integrated loudness and true peak pass.
    pub fn within_tolerance(&self) -> bool {
        self.integrated_pass && self.true_peak_pass
    }

    /// Return a short status string.
    pub fn status_str(&self) -> &'static str {
        if self.within_tolerance() {
            "PASS"
        } else {
            "FAIL"
        }
    }

    /// Return the absolute deviation in LU.
    pub fn abs_deviation_lu(&self) -> f64 {
        self.deviation_lu.abs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ebu_r128_target_lkfs() {
        assert_eq!(LoudnessTarget::EbuR128Broadcast.target_lkfs(), -23.0);
    }

    #[test]
    fn test_streaming_target_lkfs() {
        assert_eq!(LoudnessTarget::StreamingPlatform.target_lkfs(), -14.0);
    }

    #[test]
    fn test_film_mix_target_lkfs() {
        assert_eq!(LoudnessTarget::FilmMix.target_lkfs(), -27.0);
    }

    #[test]
    fn test_custom_target_lkfs() {
        assert_eq!(LoudnessTarget::Custom(-18.0).target_lkfs(), -18.0);
    }

    #[test]
    fn test_max_true_peak_broadcast() {
        assert_eq!(LoudnessTarget::EbuR128Broadcast.max_true_peak_dbtp(), -1.0);
    }

    #[test]
    fn test_max_true_peak_film() {
        assert_eq!(LoudnessTarget::FilmMix.max_true_peak_dbtp(), -2.0);
    }

    #[test]
    fn test_config_allowed_range_broadcast() {
        let cfg = LoudnessTargetConfig::new(LoudnessTarget::EbuR128Broadcast);
        let (min, max) = cfg.allowed_range();
        assert!((min - -24.0).abs() < 1e-9);
        assert!((max - -22.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_bitrate_kbps() {
        let cfg = LoudnessTargetConfig::new(LoudnessTarget::EbuR128Broadcast);
        assert_eq!(cfg.bitrate_kbps(), 320);
    }

    #[test]
    fn test_tolerance_ok_pass() {
        let cfg = LoudnessTargetConfig::new(LoudnessTarget::EbuR128Broadcast);
        assert!(cfg.tolerance_ok(-23.0));
        assert!(cfg.tolerance_ok(-23.5));
        assert!(cfg.tolerance_ok(-22.5));
    }

    #[test]
    fn test_tolerance_ok_fail() {
        let cfg = LoudnessTargetConfig::new(LoudnessTarget::EbuR128Broadcast);
        assert!(!cfg.tolerance_ok(-25.0));
        assert!(!cfg.tolerance_ok(-20.0));
    }

    #[test]
    fn test_report_within_tolerance_pass() {
        let cfg = LoudnessTargetConfig::new(LoudnessTarget::EbuR128Broadcast);
        let report = LoudnessTargetReport::new(cfg, -23.0, Some(-2.0));
        assert!(report.within_tolerance());
        assert_eq!(report.status_str(), "PASS");
    }

    #[test]
    fn test_report_within_tolerance_fail_loudness() {
        let cfg = LoudnessTargetConfig::new(LoudnessTarget::EbuR128Broadcast);
        let report = LoudnessTargetReport::new(cfg, -20.0, Some(-2.0));
        assert!(!report.within_tolerance());
        assert_eq!(report.status_str(), "FAIL");
    }

    #[test]
    fn test_report_true_peak_fail() {
        let cfg = LoudnessTargetConfig::new(LoudnessTarget::EbuR128Broadcast);
        let report = LoudnessTargetReport::new(cfg, -23.0, Some(0.5)); // 0.5 dBTP > -1.0
        assert!(!report.within_tolerance());
    }

    #[test]
    fn test_report_deviation() {
        let cfg = LoudnessTargetConfig::new(LoudnessTarget::EbuR128Broadcast);
        let report = LoudnessTargetReport::new(cfg, -21.0, None);
        assert!((report.deviation_lu - 2.0).abs() < 1e-9);
        assert!((report.abs_deviation_lu() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_strict_config_narrower_range() {
        let strict = LoudnessTargetConfig::strict(LoudnessTarget::EbuR128Broadcast);
        let (min, max) = strict.allowed_range();
        assert!((min - -23.5).abs() < 1e-9);
        assert!((max - -22.5).abs() < 1e-9);
    }

    #[test]
    fn test_target_label() {
        assert_eq!(
            LoudnessTarget::EbuR128Broadcast.label(),
            "EBU R128 Broadcast"
        );
        assert_eq!(
            LoudnessTarget::StreamingPlatform.label(),
            "Streaming Platform"
        );
        assert_eq!(LoudnessTarget::FilmMix.label(), "Film Mix");
        assert_eq!(LoudnessTarget::Custom(-18.0).label(), "Custom");
    }
}
