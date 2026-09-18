//! Audio loudness normalization for broadcast and streaming compliance.

use crate::{Result, TranscodeError};
use serde::{Deserialize, Serialize};

/// Loudness measurement standards.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoudnessStandard {
    /// EBU R128 (European Broadcasting Union) - Target: -23 LUFS.
    EbuR128,
    /// ATSC A/85 (US broadcast) - Target: -24 LKFS.
    AtscA85,
    /// Apple iTunes/Apple Music - Target: -16 LUFS.
    AppleMusic,
    /// Spotify - Target: -14 LUFS.
    Spotify,
    /// `YouTube` - Target: -13 to -15 LUFS.
    YouTube,
    /// Amazon Music - Target: -14 LUFS.
    Amazon,
    /// Tidal - Target: -14 LUFS.
    Tidal,
    /// Deezer - Target: -15 LUFS.
    Deezer,
    /// Custom target loudness.
    Custom(i32),
}

/// Loudness target configuration.
#[derive(Debug, Clone)]
pub struct LoudnessTarget {
    /// Target integrated loudness in LUFS/LKFS.
    pub target_lufs: f64,
    /// Maximum true peak level in dBTP.
    pub max_true_peak_dbtp: f64,
    /// Loudness range tolerance in LU.
    pub loudness_range: Option<(f64, f64)>,
    /// Whether to measure loudness only (no normalization).
    pub measure_only: bool,
}

impl LoudnessStandard {
    /// Gets the target loudness in LUFS/LKFS.
    #[must_use]
    pub fn target_lufs(self) -> f64 {
        match self {
            Self::EbuR128 => -23.0,
            Self::AtscA85 => -24.0,
            Self::AppleMusic => -16.0,
            Self::Spotify => -14.0,
            Self::YouTube => -14.0,
            Self::Amazon => -14.0,
            Self::Tidal => -14.0,
            Self::Deezer => -15.0,
            Self::Custom(lufs) => f64::from(lufs),
        }
    }

    /// Gets the maximum true peak level in dBTP.
    #[must_use]
    pub fn max_true_peak_dbtp(self) -> f64 {
        match self {
            Self::EbuR128 => -1.0,
            Self::AtscA85 => -2.0,
            Self::AppleMusic => -1.0,
            Self::Spotify => -2.0,
            Self::YouTube => -1.0,
            Self::Amazon => -2.0,
            Self::Tidal => -1.0,
            Self::Deezer => -1.0,
            Self::Custom(_) => -1.0,
        }
    }

    /// Gets a human-readable description of the standard.
    #[must_use]
    pub fn description(self) -> &'static str {
        match self {
            Self::EbuR128 => "EBU R128 (European broadcast standard)",
            Self::AtscA85 => "ATSC A/85 (US broadcast standard)",
            Self::AppleMusic => "Apple Music/iTunes",
            Self::Spotify => "Spotify",
            Self::YouTube => "YouTube",
            Self::Amazon => "Amazon Music",
            Self::Tidal => "Tidal",
            Self::Deezer => "Deezer",
            Self::Custom(_) => "Custom loudness target",
        }
    }

    /// Converts to a loudness target configuration.
    #[must_use]
    pub fn to_target(self) -> LoudnessTarget {
        LoudnessTarget {
            target_lufs: self.target_lufs(),
            max_true_peak_dbtp: self.max_true_peak_dbtp(),
            loudness_range: None,
            measure_only: false,
        }
    }
}

impl LoudnessTarget {
    /// Creates a new loudness target with specified LUFS.
    #[must_use]
    pub fn new(target_lufs: f64) -> Self {
        Self {
            target_lufs,
            max_true_peak_dbtp: -1.0,
            loudness_range: None,
            measure_only: false,
        }
    }

    /// Sets the maximum true peak level.
    #[must_use]
    pub fn with_max_true_peak(mut self, dbtp: f64) -> Self {
        self.max_true_peak_dbtp = dbtp;
        self
    }

    /// Sets the loudness range tolerance.
    #[must_use]
    pub fn with_loudness_range(mut self, min: f64, max: f64) -> Self {
        self.loudness_range = Some((min, max));
        self
    }

    /// Sets measure-only mode (no normalization applied).
    #[must_use]
    pub fn measure_only(mut self) -> Self {
        self.measure_only = true;
        self
    }

    /// Validates the loudness target configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub fn validate(&self) -> Result<()> {
        if self.target_lufs > 0.0 {
            return Err(TranscodeError::NormalizationError(
                "Target LUFS must be negative".to_string(),
            ));
        }

        if self.target_lufs < -70.0 {
            return Err(TranscodeError::NormalizationError(
                "Target LUFS too low (< -70 LUFS)".to_string(),
            ));
        }

        if self.max_true_peak_dbtp > 0.0 {
            return Err(TranscodeError::NormalizationError(
                "Maximum true peak must be negative or zero".to_string(),
            ));
        }

        if let Some((min, max)) = self.loudness_range {
            if min >= max {
                return Err(TranscodeError::NormalizationError(
                    "Invalid loudness range: min must be less than max".to_string(),
                ));
            }
        }

        Ok(())
    }
}

/// Normalization configuration.
#[derive(Debug, Clone)]
pub struct NormalizationConfig {
    /// Loudness standard to use.
    pub standard: LoudnessStandard,
    /// Target configuration.
    pub target: LoudnessTarget,
    /// Enable two-pass normalization for better accuracy.
    pub two_pass: bool,
    /// Enable linear gain only (no dynamic range compression).
    pub linear_only: bool,
    /// Gate threshold for loudness measurement in LUFS.
    pub gate_threshold: f64,
}

impl NormalizationConfig {
    /// Creates a new normalization config with the specified standard.
    #[must_use]
    pub fn new(standard: LoudnessStandard) -> Self {
        Self {
            standard,
            target: standard.to_target(),
            two_pass: true,
            linear_only: true,
            gate_threshold: -70.0,
        }
    }

    /// Sets two-pass mode.
    #[must_use]
    pub fn with_two_pass(mut self, enable: bool) -> Self {
        self.two_pass = enable;
        self
    }

    /// Sets linear-only mode.
    #[must_use]
    pub fn with_linear_only(mut self, enable: bool) -> Self {
        self.linear_only = enable;
        self
    }

    /// Sets the gate threshold.
    #[must_use]
    pub fn with_gate_threshold(mut self, threshold: f64) -> Self {
        self.gate_threshold = threshold;
        self
    }

    /// Validates the configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub fn validate(&self) -> Result<()> {
        self.target.validate()
    }
}

impl Default for NormalizationConfig {
    fn default() -> Self {
        Self::new(LoudnessStandard::EbuR128)
    }
}

/// Audio normalizer for applying loudness normalization.
pub struct AudioNormalizer {
    config: NormalizationConfig,
}

impl AudioNormalizer {
    /// Creates a new audio normalizer with the specified configuration.
    #[must_use]
    pub fn new(config: NormalizationConfig) -> Self {
        Self { config }
    }

    /// Creates a normalizer with the specified standard.
    #[must_use]
    pub fn with_standard(standard: LoudnessStandard) -> Self {
        Self::new(NormalizationConfig::new(standard))
    }

    /// Gets the target loudness in LUFS.
    #[must_use]
    pub fn target_lufs(&self) -> f64 {
        self.config.target.target_lufs
    }

    /// Gets the maximum true peak in dBTP.
    #[must_use]
    pub fn max_true_peak_dbtp(&self) -> f64 {
        self.config.target.max_true_peak_dbtp
    }

    /// Calculates the gain adjustment needed for normalization.
    ///
    /// # Arguments
    ///
    /// * `measured_lufs` - The measured integrated loudness
    /// * `measured_peak_dbtp` - The measured true peak
    ///
    /// # Returns
    ///
    /// The gain adjustment in dB, limited to prevent clipping.
    #[must_use]
    pub fn calculate_gain(&self, measured_lufs: f64, measured_peak_dbtp: f64) -> f64 {
        let target = self.target_lufs();
        let max_peak = self.max_true_peak_dbtp();

        // Calculate gain needed to reach target loudness
        let loudness_gain = target - measured_lufs;

        // Calculate maximum gain that won't exceed peak limit
        let peak_gain = max_peak - measured_peak_dbtp;

        // Use the more conservative (smaller) gain
        loudness_gain.min(peak_gain)
    }

    /// Checks if normalization is needed.
    ///
    /// # Arguments
    ///
    /// * `measured_lufs` - The measured integrated loudness
    /// * `tolerance` - Tolerance in LU (default: 0.5)
    #[must_use]
    pub fn needs_normalization(&self, measured_lufs: f64, tolerance: f64) -> bool {
        let diff = (measured_lufs - self.target_lufs()).abs();
        diff > tolerance
    }

    /// Gets the filter string for loudness normalization.
    ///
    /// This generates the filter parameters for audio processing.
    #[must_use]
    pub fn get_filter_string(&self) -> String {
        let target = self.target_lufs();
        let max_peak = self.max_true_peak_dbtp();

        if self.config.two_pass {
            format!("loudnorm=I={target}:TP={max_peak}:LRA=11:dual_mono=true")
        } else {
            format!("loudnorm=I={target}:TP={max_peak}")
        }
    }
}

/// Measured loudness metrics.
#[derive(Debug, Clone)]
pub struct LoudnessMetrics {
    /// Integrated loudness in LUFS.
    pub integrated_lufs: f64,
    /// Loudness range in LU.
    #[allow(dead_code)]
    pub loudness_range: f64,
    /// Maximum true peak in dBTP.
    pub true_peak_dbtp: f64,
    /// Maximum momentary loudness in LUFS.
    #[allow(dead_code)]
    pub momentary_max: f64,
    /// Maximum short-term loudness in LUFS.
    #[allow(dead_code)]
    pub short_term_max: f64,
}

impl LoudnessMetrics {
    /// Checks if the metrics are compliant with a given standard.
    #[must_use]
    #[allow(dead_code)]
    pub fn is_compliant(&self, standard: LoudnessStandard, tolerance: f64) -> bool {
        let target = standard.target_lufs();
        let max_peak = standard.max_true_peak_dbtp();

        let loudness_ok = (self.integrated_lufs - target).abs() <= tolerance;
        let peak_ok = self.true_peak_dbtp <= max_peak;

        loudness_ok && peak_ok
    }

    /// Gets a compliance report as a string.
    #[must_use]
    #[allow(dead_code)]
    pub fn compliance_report(&self, standard: LoudnessStandard) -> String {
        let target = standard.target_lufs();
        let max_peak = standard.max_true_peak_dbtp();

        format!(
            "Integrated: {:.1} LUFS (target: {:.1} LUFS)\n\
             True Peak: {:.1} dBTP (max: {:.1} dBTP)\n\
             Loudness Range: {:.1} LU",
            self.integrated_lufs, target, self.true_peak_dbtp, max_peak, self.loudness_range
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loudness_standard_targets() {
        assert_eq!(LoudnessStandard::EbuR128.target_lufs(), -23.0);
        assert_eq!(LoudnessStandard::AtscA85.target_lufs(), -24.0);
        assert_eq!(LoudnessStandard::Spotify.target_lufs(), -14.0);
        assert_eq!(LoudnessStandard::YouTube.target_lufs(), -14.0);
    }

    #[test]
    fn test_loudness_standard_peaks() {
        assert_eq!(LoudnessStandard::EbuR128.max_true_peak_dbtp(), -1.0);
        assert_eq!(LoudnessStandard::AtscA85.max_true_peak_dbtp(), -2.0);
    }

    #[test]
    fn test_custom_standard() {
        let custom = LoudnessStandard::Custom(-18);
        assert_eq!(custom.target_lufs(), -18.0);
    }

    #[test]
    fn test_loudness_target_validation() {
        let valid = LoudnessTarget::new(-23.0);
        assert!(valid.validate().is_ok());

        let invalid_positive = LoudnessTarget::new(5.0);
        assert!(invalid_positive.validate().is_err());

        let invalid_too_low = LoudnessTarget::new(-80.0);
        assert!(invalid_too_low.validate().is_err());
    }

    #[test]
    fn test_normalizer_gain_calculation() {
        let normalizer = AudioNormalizer::with_standard(LoudnessStandard::EbuR128);

        // Audio at -20 LUFS, peak at -5 dBTP
        // Target: -23 LUFS, max peak: -1 dBTP
        let gain = normalizer.calculate_gain(-20.0, -5.0);

        // Loudness gain would be -3.0 dB (to go from -20 to -23)
        // Peak gain would be +4.0 dB (to go from -5 to -1)
        // Should use loudness gain (more conservative)
        assert_eq!(gain, -3.0);
    }

    #[test]
    fn test_normalizer_needs_normalization() {
        let normalizer = AudioNormalizer::with_standard(LoudnessStandard::EbuR128);

        assert!(!normalizer.needs_normalization(-23.0, 0.5)); // Exact match
        assert!(!normalizer.needs_normalization(-23.3, 0.5)); // Within tolerance
        assert!(normalizer.needs_normalization(-20.0, 0.5)); // Outside tolerance
    }

    #[test]
    fn test_normalizer_filter_string() {
        let normalizer = AudioNormalizer::with_standard(LoudnessStandard::EbuR128);
        let filter = normalizer.get_filter_string();

        assert!(filter.contains("loudnorm"));
        assert!(filter.contains("I=-23"));
        assert!(filter.contains("TP=-1"));
    }

    #[test]
    fn test_loudness_metrics_compliance() {
        let metrics = LoudnessMetrics {
            integrated_lufs: -23.2,
            loudness_range: 8.0,
            true_peak_dbtp: -1.5,
            momentary_max: -15.0,
            short_term_max: -18.0,
        };

        assert!(metrics.is_compliant(LoudnessStandard::EbuR128, 0.5));
    }

    #[test]
    fn test_loudness_metrics_non_compliant() {
        let metrics = LoudnessMetrics {
            integrated_lufs: -18.0, // Too loud
            loudness_range: 8.0,
            true_peak_dbtp: -1.5,
            momentary_max: -15.0,
            short_term_max: -18.0,
        };

        assert!(!metrics.is_compliant(LoudnessStandard::EbuR128, 0.5));
    }

    #[test]
    fn test_normalization_config_builder() {
        let config = NormalizationConfig::new(LoudnessStandard::Spotify)
            .with_two_pass(true)
            .with_linear_only(false)
            .with_gate_threshold(-50.0);

        assert_eq!(config.standard, LoudnessStandard::Spotify);
        assert!(config.two_pass);
        assert!(!config.linear_only);
        assert_eq!(config.gate_threshold, -50.0);
    }
}
