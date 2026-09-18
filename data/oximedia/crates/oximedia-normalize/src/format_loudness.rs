#![allow(dead_code)]
//! Format-specific loudness compliance checking for broadcast standards.

/// Broadcast/streaming loudness formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BroadcastFormat {
    /// ATSC A/85 — US broadcast standard.
    AtscA85,
    /// EBU R128 — European broadcast standard.
    EbuR128,
    /// ARIB TR-B32 — Japanese broadcast standard.
    AribTrB32,
    /// Netflix loudness spec (-27 LUFS streaming drama / -24 LUFS loud).
    NetflixDrama,
    /// Spotify — -14 LUFS.
    Spotify,
    /// YouTube — -14 LUFS integrated.
    YouTube,
    /// Apple Music — -16 LUFS.
    AppleMusic,
    /// Amazon Music — -14 LUFS.
    AmazonMusic,
}

impl BroadcastFormat {
    /// Target integrated loudness in LKFS/LUFS.
    pub fn target_lkfs(&self) -> f64 {
        match self {
            Self::AtscA85 => -24.0,
            Self::EbuR128 => -23.0,
            Self::AribTrB32 => -24.0,
            Self::NetflixDrama => -27.0,
            Self::Spotify => -14.0,
            Self::YouTube => -14.0,
            Self::AppleMusic => -16.0,
            Self::AmazonMusic => -14.0,
        }
    }

    /// Maximum allowed true peak in dBTP.
    pub fn max_true_peak_dbtp(&self) -> f64 {
        match self {
            Self::AtscA85 => -2.0,
            Self::EbuR128 => -1.0,
            Self::AribTrB32 => -2.0,
            Self::NetflixDrama => -2.0,
            Self::Spotify => -1.0,
            Self::YouTube => -1.0,
            Self::AppleMusic => -1.0,
            Self::AmazonMusic => -2.0,
        }
    }

    /// Tolerance around target (±LU).
    pub fn tolerance_lu(&self) -> f64 {
        match self {
            Self::AtscA85 => 2.0,
            Self::EbuR128 => 1.0,
            Self::AribTrB32 => 2.0,
            Self::NetflixDrama => 1.0,
            Self::Spotify => 1.0,
            Self::YouTube => 1.0,
            Self::AppleMusic => 1.0,
            Self::AmazonMusic => 1.0,
        }
    }

    /// Human-readable name of the format.
    pub fn name(&self) -> &'static str {
        match self {
            Self::AtscA85 => "ATSC A/85",
            Self::EbuR128 => "EBU R128",
            Self::AribTrB32 => "ARIB TR-B32",
            Self::NetflixDrama => "Netflix Drama",
            Self::Spotify => "Spotify",
            Self::YouTube => "YouTube",
            Self::AppleMusic => "Apple Music",
            Self::AmazonMusic => "Amazon Music",
        }
    }

    /// Whether this is a broadcast (as opposed to streaming) format.
    pub fn is_broadcast(&self) -> bool {
        matches!(self, Self::AtscA85 | Self::EbuR128 | Self::AribTrB32)
    }
}

/// Measured loudness for a format compliance check.
#[derive(Debug, Clone)]
pub struct FormatLoudness {
    /// Integrated loudness (LKFS/LUFS).
    pub integrated_lkfs: f64,
    /// Maximum true peak (dBTP).
    pub max_true_peak_dbtp: f64,
    /// Loudness range in LU.
    pub loudness_range_lu: f64,
}

impl FormatLoudness {
    /// Create a new measurement.
    pub fn new(integrated_lkfs: f64, max_true_peak_dbtp: f64, loudness_range_lu: f64) -> Self {
        Self {
            integrated_lkfs,
            max_true_peak_dbtp,
            loudness_range_lu,
        }
    }

    /// Check whether the measurement meets the given broadcast format spec.
    pub fn meets_spec(&self, format: BroadcastFormat) -> bool {
        let lkfs_ok = (self.integrated_lkfs - format.target_lkfs()).abs() <= format.tolerance_lu();
        let peak_ok = self.max_true_peak_dbtp <= format.max_true_peak_dbtp();
        lkfs_ok && peak_ok
    }

    /// Gain needed to reach the format's target (positive = boost).
    pub fn gain_to_target(&self, format: BroadcastFormat) -> f64 {
        format.target_lkfs() - self.integrated_lkfs
    }

    /// Headroom before hitting the true peak ceiling.
    pub fn peak_headroom(&self, format: BroadcastFormat) -> f64 {
        format.max_true_peak_dbtp() - self.max_true_peak_dbtp
    }
}

/// Result of a format loudness check.
#[derive(Debug, Clone)]
pub struct FormatLoudnessCheckResult {
    /// Format checked against.
    pub format: BroadcastFormat,
    /// The measured values.
    pub measurement: FormatLoudness,
    /// Whether integrated loudness is within spec.
    pub lkfs_pass: bool,
    /// Whether true peak is within spec.
    pub peak_pass: bool,
    /// Gain correction needed (dB).
    pub correction_db: f64,
}

impl FormatLoudnessCheckResult {
    /// Overall pass/fail.
    pub fn passes(&self) -> bool {
        self.lkfs_pass && self.peak_pass
    }
}

/// Checks audio measurements against broadcast format specifications.
#[derive(Debug, Clone)]
pub struct FormatLoudnessChecker {
    /// Format to check against.
    pub format: BroadcastFormat,
}

impl FormatLoudnessChecker {
    /// Create a checker for the given format.
    pub fn new(format: BroadcastFormat) -> Self {
        Self { format }
    }

    /// Run the check and return a detailed result.
    pub fn check(&self, measurement: FormatLoudness) -> FormatLoudnessCheckResult {
        let lkfs_pass = (measurement.integrated_lkfs - self.format.target_lkfs()).abs()
            <= self.format.tolerance_lu();
        let peak_pass = measurement.max_true_peak_dbtp <= self.format.max_true_peak_dbtp();
        let correction_db = measurement.gain_to_target(self.format);

        FormatLoudnessCheckResult {
            format: self.format,
            measurement,
            lkfs_pass,
            peak_pass,
            correction_db,
        }
    }

    /// Check against multiple formats simultaneously.
    pub fn check_all(
        &self,
        measurement: &FormatLoudness,
        formats: &[BroadcastFormat],
    ) -> Vec<FormatLoudnessCheckResult> {
        formats
            .iter()
            .map(|&fmt| {
                let checker = FormatLoudnessChecker::new(fmt);
                checker.check(measurement.clone())
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atsc_target_lkfs() {
        assert!((BroadcastFormat::AtscA85.target_lkfs() - (-24.0)).abs() < 1e-9);
    }

    #[test]
    fn test_ebu_target_lkfs() {
        assert!((BroadcastFormat::EbuR128.target_lkfs() - (-23.0)).abs() < 1e-9);
    }

    #[test]
    fn test_spotify_target_lkfs() {
        assert!((BroadcastFormat::Spotify.target_lkfs() - (-14.0)).abs() < 1e-9);
    }

    #[test]
    fn test_max_true_peak_ebu() {
        assert!((BroadcastFormat::EbuR128.max_true_peak_dbtp() - (-1.0)).abs() < 1e-9);
    }

    #[test]
    fn test_format_is_broadcast() {
        assert!(BroadcastFormat::AtscA85.is_broadcast());
        assert!(BroadcastFormat::EbuR128.is_broadcast());
        assert!(!BroadcastFormat::Spotify.is_broadcast());
        assert!(!BroadcastFormat::YouTube.is_broadcast());
    }

    #[test]
    fn test_format_name() {
        assert_eq!(BroadcastFormat::EbuR128.name(), "EBU R128");
        assert_eq!(BroadcastFormat::AtscA85.name(), "ATSC A/85");
        assert_eq!(BroadcastFormat::AribTrB32.name(), "ARIB TR-B32");
    }

    #[test]
    fn test_meets_spec_pass() {
        let m = FormatLoudness::new(-24.0, -3.0, 8.0);
        assert!(m.meets_spec(BroadcastFormat::AtscA85));
    }

    #[test]
    fn test_meets_spec_fail_lkfs() {
        let m = FormatLoudness::new(-30.0, -3.0, 8.0); // 6 LU below target
        assert!(!m.meets_spec(BroadcastFormat::AtscA85));
    }

    #[test]
    fn test_meets_spec_fail_peak() {
        let m = FormatLoudness::new(-23.0, -0.5, 8.0); // peak above ceiling
        assert!(!m.meets_spec(BroadcastFormat::EbuR128));
    }

    #[test]
    fn test_gain_to_target() {
        let m = FormatLoudness::new(-20.0, -2.0, 5.0);
        let gain = m.gain_to_target(BroadcastFormat::EbuR128);
        assert!((gain - (-3.0)).abs() < 1e-9); // -23 - (-20) = -3
    }

    #[test]
    fn test_peak_headroom() {
        let m = FormatLoudness::new(-23.0, -3.0, 6.0);
        let headroom = m.peak_headroom(BroadcastFormat::EbuR128);
        assert!((headroom - 2.0).abs() < 1e-9); // -1 - (-3) = 2
    }

    #[test]
    fn test_checker_check_pass() {
        let checker = FormatLoudnessChecker::new(BroadcastFormat::AtscA85);
        let m = FormatLoudness::new(-24.0, -3.0, 7.0);
        let result = checker.check(m);
        assert!(result.passes());
    }

    #[test]
    fn test_checker_check_fail_peak() {
        let checker = FormatLoudnessChecker::new(BroadcastFormat::EbuR128);
        let m = FormatLoudness::new(-23.0, -0.3, 6.0);
        let result = checker.check(m);
        assert!(!result.peak_pass);
        assert!(!result.passes());
    }

    #[test]
    fn test_checker_correction_db() {
        let checker = FormatLoudnessChecker::new(BroadcastFormat::EbuR128);
        let m = FormatLoudness::new(-28.0, -5.0, 5.0);
        let result = checker.check(m);
        assert!((result.correction_db - 5.0).abs() < 1e-9); // -23 - (-28) = 5
    }

    #[test]
    fn test_check_all_multiple_formats() {
        let checker = FormatLoudnessChecker::new(BroadcastFormat::EbuR128);
        let m = FormatLoudness::new(-14.0, -2.0, 5.0);
        let formats = [BroadcastFormat::Spotify, BroadcastFormat::YouTube];
        let results = checker.check_all(&m, &formats);
        assert_eq!(results.len(), 2);
        // -14 LUFS exactly matches Spotify and YouTube targets.
        assert!(results[0].passes());
        assert!(results[1].passes());
    }
}
