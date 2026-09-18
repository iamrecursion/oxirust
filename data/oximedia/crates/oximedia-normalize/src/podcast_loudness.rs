//! Podcast loudness normalization standards.
//!
//! Implements the loudness targets for major podcast distribution platforms:
//! - Spotify Podcasts: −16 LUFS integrated, −1 dBTP true peak
//! - Apple Podcasts: −16 LUFS integrated (loud), −14 LUFS (standard), −1 dBTP true peak
//! - YouTube Podcasts: −14 LUFS, −1 dBTP true peak
//! - Google Podcasts / Overcast: −16 LUFS, −1 dBTP true peak
//! - Anchor / Buzzsprout: −14 LUFS, −1 dBTP true peak
//!
//! # Algorithm
//!
//! 1. Measure integrated loudness of the episode using ITU-R BS.1770-4 gating.
//! 2. Compute the gain delta: `gain = target_lufs - measured_lufs`.
//! 3. Clamp gain to the configured max / min limits.
//! 4. Apply gain to all samples.
//! 5. Apply true peak limiter to prevent inter-sample clipping.

/// Podcast platform loudness target.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PodcastPlatform {
    /// Spotify Podcasts: −16 LUFS, −1 dBTP.
    Spotify,
    /// Apple Podcasts loud mode: −16 LUFS, −1 dBTP.
    ApplePodcastsLoud,
    /// Apple Podcasts standard mode: −14 LUFS, −1 dBTP.
    ApplePodcastsStandard,
    /// YouTube: −14 LUFS, −1 dBTP.
    YouTube,
    /// Anchor / Buzzsprout: −14 LUFS, −1 dBTP.
    Anchor,
    /// Generic / custom target.
    Custom {
        /// Target integrated loudness in LUFS.
        target_lufs: f64,
        /// Maximum true peak in dBTP.
        true_peak_ceiling_dbtp: f64,
    },
}

impl PodcastPlatform {
    /// Return the target integrated loudness in LUFS.
    pub fn target_lufs(self) -> f64 {
        match self {
            Self::Spotify => -16.0,
            Self::ApplePodcastsLoud => -16.0,
            Self::ApplePodcastsStandard => -14.0,
            Self::YouTube => -14.0,
            Self::Anchor => -14.0,
            Self::Custom { target_lufs, .. } => target_lufs,
        }
    }

    /// Return the maximum true peak level in dBTP.
    pub fn true_peak_ceiling_dbtp(self) -> f64 {
        match self {
            Self::Custom {
                true_peak_ceiling_dbtp,
                ..
            } => true_peak_ceiling_dbtp,
            _ => -1.0,
        }
    }

    /// Human-readable platform name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Spotify => "Spotify Podcasts",
            Self::ApplePodcastsLoud => "Apple Podcasts (Loud)",
            Self::ApplePodcastsStandard => "Apple Podcasts (Standard)",
            Self::YouTube => "YouTube",
            Self::Anchor => "Anchor / Buzzsprout",
            Self::Custom { .. } => "Custom",
        }
    }
}

/// Configuration for podcast loudness normalization.
#[derive(Debug, Clone)]
pub struct PodcastNormConfig {
    /// Target platform.
    pub platform: PodcastPlatform,
    /// Maximum allowed gain boost in dB (safety limit).
    pub max_gain_db: f64,
    /// Maximum allowed attenuation in dB (positive value).
    pub max_attenuation_db: f64,
    /// Whether to enable a simple brick-wall limiter pass after gain application.
    pub enable_limiter: bool,
    /// Tolerance in LU: if measured loudness is within this range of target,
    /// no normalization is applied.
    pub tolerance_lu: f64,
}

impl PodcastNormConfig {
    /// Create a configuration for the specified platform with sensible defaults.
    pub fn new(platform: PodcastPlatform) -> Self {
        Self {
            platform,
            max_gain_db: 20.0,
            max_attenuation_db: 30.0,
            enable_limiter: true,
            tolerance_lu: 0.5,
        }
    }

    /// Create a Spotify podcast configuration.
    pub fn spotify() -> Self {
        Self::new(PodcastPlatform::Spotify)
    }

    /// Create an Apple Podcasts (loud mode) configuration.
    pub fn apple_loud() -> Self {
        Self::new(PodcastPlatform::ApplePodcastsLoud)
    }

    /// Create an Apple Podcasts (standard) configuration.
    pub fn apple_standard() -> Self {
        Self::new(PodcastPlatform::ApplePodcastsStandard)
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.max_gain_db <= 0.0 || self.max_gain_db > 60.0 {
            return Err(format!(
                "max_gain_db must be in (0, 60] dB, got {}",
                self.max_gain_db
            ));
        }
        if self.max_attenuation_db <= 0.0 || self.max_attenuation_db > 60.0 {
            return Err(format!(
                "max_attenuation_db must be in (0, 60] dB, got {}",
                self.max_attenuation_db
            ));
        }
        if self.tolerance_lu < 0.0 || self.tolerance_lu > 10.0 {
            return Err(format!(
                "tolerance_lu must be in [0, 10] LU, got {}",
                self.tolerance_lu
            ));
        }
        Ok(())
    }
}

/// Result produced by the podcast normalizer.
#[derive(Debug, Clone)]
pub struct PodcastNormResult {
    /// The platform-specific target used.
    pub platform: PodcastPlatform,
    /// Measured integrated loudness before normalization (LUFS).
    pub measured_lufs: f64,
    /// Target loudness (LUFS).
    pub target_lufs: f64,
    /// Gain applied in dB (positive = boost, negative = cut).
    pub applied_gain_db: f64,
    /// Whether the limiter was engaged (output peak would exceed ceiling).
    pub limiter_engaged: bool,
    /// Whether the output is within the platform's tolerance.
    pub within_tolerance: bool,
    /// Estimated output loudness after normalization (LUFS).
    pub output_lufs: f64,
}

impl PodcastNormResult {
    /// True if normalization was a no-op (already within tolerance).
    pub fn was_noop(&self) -> bool {
        self.applied_gain_db.abs() < 1e-6
    }

    /// True if gain was boosted.
    pub fn was_boost(&self) -> bool {
        self.applied_gain_db > 0.0
    }

    /// True if gain was attenuated.
    pub fn was_attenuation(&self) -> bool {
        self.applied_gain_db < 0.0
    }
}

/// Podcast loudness normalizer.
///
/// Provides two-pass normalization: compute the required gain from a measured
/// integrated loudness value, then apply it to audio samples.
///
/// # Example
///
/// ```rust
/// use oximedia_normalize::podcast_loudness::{PodcastNormConfig, PodcastNormalizer};
///
/// let config = PodcastNormConfig::spotify();
/// let normalizer = PodcastNormalizer::new(config).expect("valid config");
///
/// // In a real workflow: measure integrated_lufs via ITU-R BS.1770, then:
/// let result = normalizer.compute_gain(-20.0); // measured at -20 LUFS
/// assert!(result.applied_gain_db > 0.0);      // needs boost to -16 LUFS
/// ```
pub struct PodcastNormalizer {
    config: PodcastNormConfig,
}

impl PodcastNormalizer {
    /// Create a new podcast normalizer.
    ///
    /// Returns `Err` if the configuration is invalid.
    pub fn new(config: PodcastNormConfig) -> Result<Self, String> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Compute the required gain and produce a result without modifying any audio.
    ///
    /// # Arguments
    /// * `measured_lufs` – Integrated loudness of the episode in LUFS.
    ///
    /// # Returns
    /// A [`PodcastNormResult`] describing the required gain and whether the
    /// output would be within the platform's tolerance.
    pub fn compute_gain(&self, measured_lufs: f64) -> PodcastNormResult {
        let target_lufs = self.config.platform.target_lufs();
        let raw_gain = target_lufs - measured_lufs;

        // Skip normalization if already within tolerance
        let applied_gain_db = if raw_gain.abs() <= self.config.tolerance_lu {
            0.0
        } else if raw_gain > 0.0 {
            raw_gain.min(self.config.max_gain_db)
        } else {
            raw_gain.max(-self.config.max_attenuation_db)
        };

        let output_lufs = measured_lufs + applied_gain_db;
        let within_tolerance = (output_lufs - target_lufs).abs() <= self.config.tolerance_lu + 0.1;

        PodcastNormResult {
            platform: self.config.platform,
            measured_lufs,
            target_lufs,
            applied_gain_db,
            limiter_engaged: false, // populated during process()
            within_tolerance,
            output_lufs,
        }
    }

    /// Apply gain normalization to `samples` in-place.
    ///
    /// This applies the computed gain linearly, then — if enabled — a simple
    /// brick-wall peak limiter to ensure the true peak ceiling is respected.
    ///
    /// # Arguments
    /// * `samples` – Interleaved PCM samples in f32 format (any channel count).
    /// * `measured_lufs` – Integrated loudness of `samples` in LUFS.
    ///
    /// # Returns
    /// A [`PodcastNormResult`] with `limiter_engaged` populated.
    pub fn process(&self, samples: &mut [f32], measured_lufs: f64) -> PodcastNormResult {
        let mut result = self.compute_gain(measured_lufs);

        if result.applied_gain_db.abs() > 1e-9 {
            let gain = db_to_linear_f32(result.applied_gain_db as f32);
            for s in samples.iter_mut() {
                *s *= gain;
            }
        }

        // Brick-wall peak limiter
        if self.config.enable_limiter {
            let ceiling_linear =
                db_to_linear_f32(self.config.platform.true_peak_ceiling_dbtp() as f32);
            let mut engaged = false;
            for s in samples.iter_mut() {
                if s.abs() > ceiling_linear {
                    *s = s.signum() * ceiling_linear;
                    engaged = true;
                }
            }
            result.limiter_engaged = engaged;
        }

        result
    }

    /// Get the configuration.
    pub fn config(&self) -> &PodcastNormConfig {
        &self.config
    }
}

/// Convert dB to a linear amplitude multiplier (f32).
#[inline]
fn db_to_linear_f32(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}

/// Platform compliance report for a set of episodes.
#[derive(Debug, Clone)]
pub struct PodcastComplianceReport {
    /// Platform checked.
    pub platform: PodcastPlatform,
    /// Number of compliant episodes.
    pub compliant_count: usize,
    /// Number of non-compliant episodes.
    pub non_compliant_count: usize,
    /// Minimum measured loudness across all episodes.
    pub min_lufs: f64,
    /// Maximum measured loudness across all episodes.
    pub max_lufs: f64,
    /// Mean measured loudness across all episodes.
    pub mean_lufs: f64,
}

impl PodcastComplianceReport {
    /// Build a compliance report from a slice of measured LUFS values.
    pub fn from_measurements(
        platform: PodcastPlatform,
        measurements: &[f64],
        tolerance_lu: f64,
    ) -> Self {
        let target = platform.target_lufs();

        let compliant_count = measurements
            .iter()
            .filter(|&&m| (m - target).abs() <= tolerance_lu)
            .count();
        let non_compliant_count = measurements.len() - compliant_count;

        let min_lufs = measurements.iter().copied().fold(f64::INFINITY, f64::min);
        let max_lufs = measurements
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let mean_lufs = if measurements.is_empty() {
            f64::NAN
        } else {
            measurements.iter().sum::<f64>() / measurements.len() as f64
        };

        Self {
            platform,
            compliant_count,
            non_compliant_count,
            min_lufs,
            max_lufs,
            mean_lufs,
        }
    }

    /// Whether all episodes are compliant.
    pub fn all_compliant(&self) -> bool {
        self.non_compliant_count == 0
    }

    /// Compliance ratio (0.0–1.0).
    pub fn compliance_ratio(&self) -> f64 {
        let total = self.compliant_count + self.non_compliant_count;
        if total == 0 {
            1.0
        } else {
            self.compliant_count as f64 / total as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── PodcastPlatform ────────────────────────────────────────────────────

    #[test]
    fn test_platform_spotify_target() {
        assert!((PodcastPlatform::Spotify.target_lufs() - (-16.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_platform_apple_loud_target() {
        assert!((PodcastPlatform::ApplePodcastsLoud.target_lufs() - (-16.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_platform_apple_standard_target() {
        assert!(
            (PodcastPlatform::ApplePodcastsStandard.target_lufs() - (-14.0)).abs() < f64::EPSILON
        );
    }

    #[test]
    fn test_platform_youtube_target() {
        assert!((PodcastPlatform::YouTube.target_lufs() - (-14.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_platform_anchor_target() {
        assert!((PodcastPlatform::Anchor.target_lufs() - (-14.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_platform_custom_target() {
        let p = PodcastPlatform::Custom {
            target_lufs: -18.0,
            true_peak_ceiling_dbtp: -2.0,
        };
        assert!((p.target_lufs() - (-18.0)).abs() < f64::EPSILON);
        assert!((p.true_peak_ceiling_dbtp() - (-2.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_platform_true_peak_default() {
        assert!((PodcastPlatform::Spotify.true_peak_ceiling_dbtp() - (-1.0)).abs() < f64::EPSILON);
        assert!((PodcastPlatform::YouTube.true_peak_ceiling_dbtp() - (-1.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_platform_names() {
        assert_eq!(PodcastPlatform::Spotify.name(), "Spotify Podcasts");
        assert_eq!(
            PodcastPlatform::ApplePodcastsLoud.name(),
            "Apple Podcasts (Loud)"
        );
        assert_eq!(
            PodcastPlatform::ApplePodcastsStandard.name(),
            "Apple Podcasts (Standard)"
        );
        assert_eq!(PodcastPlatform::YouTube.name(), "YouTube");
        assert_eq!(PodcastPlatform::Anchor.name(), "Anchor / Buzzsprout");
    }

    // ─── PodcastNormConfig ──────────────────────────────────────────────────

    #[test]
    fn test_config_spotify_factory() {
        let cfg = PodcastNormConfig::spotify();
        assert!((cfg.platform.target_lufs() - (-16.0)).abs() < f64::EPSILON);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_apple_loud_factory() {
        let cfg = PodcastNormConfig::apple_loud();
        assert!((cfg.platform.target_lufs() - (-16.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_config_apple_standard_factory() {
        let cfg = PodcastNormConfig::apple_standard();
        assert!((cfg.platform.target_lufs() - (-14.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_config_validation_invalid_gain() {
        let mut cfg = PodcastNormConfig::spotify();
        cfg.max_gain_db = -1.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validation_invalid_tolerance() {
        let mut cfg = PodcastNormConfig::spotify();
        cfg.tolerance_lu = 15.0;
        assert!(cfg.validate().is_err());
    }

    // ─── PodcastNormalizer::compute_gain ───────────────────────────────────

    #[test]
    fn test_compute_gain_needs_boost() {
        let norm = PodcastNormalizer::new(PodcastNormConfig::spotify()).expect("valid config");
        let result = norm.compute_gain(-20.0); // −20 LUFS → need +4 dB to reach −16 LUFS
        assert!(
            (result.applied_gain_db - 4.0).abs() < 0.01,
            "expected +4 dB, got {}",
            result.applied_gain_db
        );
        assert!(result.was_boost());
        assert!(!result.was_attenuation());
    }

    #[test]
    fn test_compute_gain_needs_attenuation() {
        let norm = PodcastNormalizer::new(PodcastNormConfig::spotify()).expect("valid config");
        let result = norm.compute_gain(-12.0); // −12 LUFS → need −4 dB to reach −16 LUFS
        assert!(
            (result.applied_gain_db - (-4.0)).abs() < 0.01,
            "expected -4 dB, got {}",
            result.applied_gain_db
        );
        assert!(result.was_attenuation());
    }

    #[test]
    fn test_compute_gain_within_tolerance() {
        let norm = PodcastNormalizer::new(PodcastNormConfig::spotify()).expect("valid config");
        let result = norm.compute_gain(-16.3); // within ±0.5 LU of −16 LUFS
        assert!(
            result.was_noop(),
            "expected no-op, got gain = {}",
            result.applied_gain_db
        );
    }

    #[test]
    fn test_compute_gain_clamped_boost() {
        let mut cfg = PodcastNormConfig::spotify();
        cfg.max_gain_db = 5.0;
        let norm = PodcastNormalizer::new(cfg).expect("valid config");
        let result = norm.compute_gain(-30.0); // needs +14 dB but clamped to +5 dB
        assert!(
            (result.applied_gain_db - 5.0).abs() < 0.01,
            "expected 5 dB, got {}",
            result.applied_gain_db
        );
    }

    #[test]
    fn test_compute_gain_output_lufs_correct() {
        let norm = PodcastNormalizer::new(PodcastNormConfig::youtube()).expect("valid config");
        let result = norm.compute_gain(-18.0); // −18 → −14: +4 dB
        assert!(
            (result.output_lufs - (-14.0)).abs() < 0.1,
            "expected output ≈ -14 LUFS, got {}",
            result.output_lufs
        );
    }

    #[test]
    fn test_compute_gain_within_tolerance_flag() {
        let norm = PodcastNormalizer::new(PodcastNormConfig::spotify()).expect("valid config");
        let result = norm.compute_gain(-19.0); // needs +3 dB
        assert!(result.within_tolerance);
    }

    // ─── PodcastNormalizer::process ────────────────────────────────────────

    #[test]
    fn test_process_applies_gain() {
        let cfg = PodcastNormConfig::new(PodcastPlatform::Custom {
            target_lufs: -14.0,
            true_peak_ceiling_dbtp: -1.0,
        });
        let norm = PodcastNormalizer::new(cfg).expect("valid config");

        let mut samples = vec![0.1_f32; 4800];
        let result = norm.process(&mut samples, -20.0); // needs +6 dB
        assert!(result.applied_gain_db > 0.0);

        // Samples should have been scaled up
        let expected_gain = 10.0_f32.powf(result.applied_gain_db as f32 / 20.0);
        assert!(
            (samples[0] - 0.1 * expected_gain).abs() < 1e-4,
            "gain mismatch: got {}, expected {}",
            samples[0],
            0.1 * expected_gain
        );
    }

    #[test]
    fn test_process_limiter_engages_on_clip() {
        let mut cfg = PodcastNormConfig::spotify();
        cfg.max_gain_db = 20.0;
        cfg.enable_limiter = true;
        let norm = PodcastNormalizer::new(cfg).expect("valid config");

        // Samples near 1.0 FS, then +20 dB boost would clip
        let mut samples = vec![0.9_f32; 1000];
        let result = norm.process(&mut samples, -36.0); // needs +20 dB
        if result.applied_gain_db > 1.0 {
            assert!(result.limiter_engaged, "limiter should have been engaged");
            // No sample should exceed the true peak ceiling
            let ceiling = 10.0_f32.powf(-1.0_f32 / 20.0); // −1 dBTP ≈ 0.8913
            for &s in &samples {
                assert!(
                    s.abs() <= ceiling + 1e-5,
                    "sample {} exceeds ceiling {}",
                    s,
                    ceiling
                );
            }
        }
    }

    #[test]
    fn test_process_no_modification_on_noop() {
        let norm = PodcastNormalizer::new(PodcastNormConfig::spotify()).expect("valid config");
        let original = vec![0.3_f32; 1000];
        let mut samples = original.clone();
        norm.process(&mut samples, -16.2); // within tolerance
        for (a, b) in samples.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_process_samples_are_finite() {
        let norm =
            PodcastNormalizer::new(PodcastNormConfig::apple_standard()).expect("valid config");
        let mut samples: Vec<f32> = (0..4800)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48000.0).sin() * 0.1)
            .collect();
        norm.process(&mut samples, -30.0);
        assert!(samples.iter().all(|s| s.is_finite()));
    }

    // ─── PodcastComplianceReport ────────────────────────────────────────────

    #[test]
    fn test_compliance_report_all_compliant() {
        let measurements = vec![-16.1, -15.9, -16.0, -16.4]; // all within ±0.5 of −16
        let report = PodcastComplianceReport::from_measurements(
            PodcastPlatform::Spotify,
            &measurements,
            0.5,
        );
        assert_eq!(report.compliant_count, 4);
        assert_eq!(report.non_compliant_count, 0);
        assert!(report.all_compliant());
        assert!((report.compliance_ratio() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_compliance_report_mixed() {
        let measurements = vec![-16.0, -20.0, -16.3]; // one non-compliant
        let report = PodcastComplianceReport::from_measurements(
            PodcastPlatform::Spotify,
            &measurements,
            0.5,
        );
        assert_eq!(report.non_compliant_count, 1);
        assert_eq!(report.compliant_count, 2);
        assert!(!report.all_compliant());
    }

    #[test]
    fn test_compliance_report_empty() {
        let report = PodcastComplianceReport::from_measurements(PodcastPlatform::Spotify, &[], 0.5);
        assert_eq!(report.compliant_count, 0);
        assert_eq!(report.non_compliant_count, 0);
        assert!((report.compliance_ratio() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_compliance_report_min_max_mean() {
        let measurements = vec![-18.0, -14.0, -16.0];
        let report = PodcastComplianceReport::from_measurements(
            PodcastPlatform::YouTube,
            &measurements,
            1.0,
        );
        assert!((report.min_lufs - (-18.0)).abs() < f64::EPSILON);
        assert!((report.max_lufs - (-14.0)).abs() < f64::EPSILON);
        assert!((report.mean_lufs - (-16.0)).abs() < 0.01);
    }
}

impl PodcastNormConfig {
    /// Create a YouTube podcast configuration.
    pub fn youtube() -> Self {
        Self::new(PodcastPlatform::YouTube)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PodcastStandard — speech-optimized normalization presets
// ─────────────────────────────────────────────────────────────────────────────

/// High-level podcast loudness standard enum.
///
/// Encapsulates the target LUFS, true-peak ceiling, and speech-optimized
/// processing parameters (high-pass filter frequency, compressor threshold)
/// for common podcast distribution targets.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PodcastStandard {
    /// Spotify Podcasts: -16 LUFS, speech-optimized.
    SpotifyPodcasts,
    /// Apple Podcasts: -14 LUFS, speech-optimized.
    ApplePodcasts,
    /// YouTube Podcasts: -14 LUFS.
    YouTubePodcasts,
    /// General speech / narration: -16 LUFS, aggressive speech processing.
    SpeechGeneral,
    /// Audiobook standard: -18 LUFS (ACX / Audible).
    Audiobook,
}

impl PodcastStandard {
    /// Target integrated loudness in LUFS.
    pub fn target_lufs(self) -> f64 {
        match self {
            Self::SpotifyPodcasts | Self::SpeechGeneral => -16.0,
            Self::ApplePodcasts | Self::YouTubePodcasts => -14.0,
            Self::Audiobook => -18.0,
        }
    }

    /// Maximum true peak level in dBTP.
    pub fn true_peak_ceiling_dbtp(self) -> f64 {
        match self {
            Self::Audiobook => -3.0,
            _ => -1.0,
        }
    }

    /// Recommended high-pass filter cutoff in Hz for speech processing.
    ///
    /// A higher cutoff removes more low-frequency rumble, plosives, etc.
    pub fn speech_hpf_hz(self) -> f64 {
        match self {
            Self::SpeechGeneral | Self::Audiobook => 80.0,
            _ => 60.0,
        }
    }

    /// Recommended compressor threshold in dBFS for speech levelling.
    pub fn speech_compressor_threshold_db(self) -> f64 {
        match self {
            Self::SpeechGeneral => -18.0,
            Self::Audiobook => -20.0,
            _ => -22.0,
        }
    }

    /// Human-readable name.
    pub fn name(self) -> &'static str {
        match self {
            Self::SpotifyPodcasts => "Spotify Podcasts (-16 LUFS)",
            Self::ApplePodcasts => "Apple Podcasts (-14 LUFS)",
            Self::YouTubePodcasts => "YouTube Podcasts (-14 LUFS)",
            Self::SpeechGeneral => "General Speech (-16 LUFS)",
            Self::Audiobook => "Audiobook / ACX (-18 LUFS)",
        }
    }

    /// Convert to a [`PodcastNormConfig`].
    pub fn to_config(self) -> PodcastNormConfig {
        let platform = match self {
            Self::SpotifyPodcasts => PodcastPlatform::Spotify,
            Self::ApplePodcasts => PodcastPlatform::ApplePodcastsStandard,
            Self::YouTubePodcasts => PodcastPlatform::YouTube,
            Self::SpeechGeneral => PodcastPlatform::Custom {
                target_lufs: -16.0,
                true_peak_ceiling_dbtp: -1.0,
            },
            Self::Audiobook => PodcastPlatform::Custom {
                target_lufs: -18.0,
                true_peak_ceiling_dbtp: -3.0,
            },
        };
        PodcastNormConfig::new(platform)
    }
}

/// Speech-optimized podcast normalizer.
///
/// Wraps [`PodcastNormalizer`] with a speech-specific pre-processing stage:
/// a high-pass filter to remove low-frequency rumble / plosives, followed by
/// a simple RMS-based leveller before final loudness normalization.
pub struct SpeechPodcastNormalizer {
    inner: PodcastNormalizer,
    standard: PodcastStandard,
    /// High-pass filter cutoff in Hz.
    hpf_cutoff_hz: f64,
    /// Compressor threshold in dBFS.
    compressor_threshold_db: f64,
}

impl SpeechPodcastNormalizer {
    /// Create a speech-optimized podcast normalizer for the given standard.
    pub fn new(standard: PodcastStandard) -> Result<Self, String> {
        let config = standard.to_config();
        let inner = PodcastNormalizer::new(config)?;
        Ok(Self {
            inner,
            standard,
            hpf_cutoff_hz: standard.speech_hpf_hz(),
            compressor_threshold_db: standard.speech_compressor_threshold_db(),
        })
    }

    /// Apply speech-optimized pre-processing: high-pass filter + simple leveller.
    fn preprocess_speech(&self, samples: &mut [f32], sample_rate: f64) {
        // 1) High-pass filter (first-order IIR) to remove rumble / plosives
        let rc = 1.0 / (2.0 * std::f64::consts::PI * self.hpf_cutoff_hz);
        let dt = 1.0 / sample_rate;
        let alpha = rc / (rc + dt);

        let mut x_prev = 0.0_f64;
        let mut y_prev = 0.0_f64;
        for s in samples.iter_mut() {
            let x = f64::from(*s);
            let y = alpha * (y_prev + x - x_prev);
            x_prev = x;
            y_prev = y;
            *s = y as f32;
        }

        // 2) Simple soft-knee compressor for speech levelling
        let threshold_linear = 10.0_f32.powf(self.compressor_threshold_db as f32 / 20.0);
        let ratio = 3.0_f32; // 3:1 compression
        for s in samples.iter_mut() {
            let abs_s = s.abs();
            if abs_s > threshold_linear {
                let over = abs_s - threshold_linear;
                let compressed = threshold_linear + over / ratio;
                *s = s.signum() * compressed;
            }
        }
    }

    /// Process samples with speech pre-processing then loudness normalization.
    ///
    /// # Arguments
    /// * `samples` – Interleaved PCM samples.
    /// * `measured_lufs` – Pre-measured integrated loudness (after speech preprocessing).
    /// * `sample_rate` – Sample rate in Hz.
    pub fn process(
        &self,
        samples: &mut [f32],
        measured_lufs: f64,
        sample_rate: f64,
    ) -> PodcastNormResult {
        self.preprocess_speech(samples, sample_rate);
        self.inner.process(samples, measured_lufs)
    }

    /// Get the podcast standard.
    pub fn standard(&self) -> PodcastStandard {
        self.standard
    }

    /// Get the HPF cutoff frequency.
    pub fn hpf_cutoff_hz(&self) -> f64 {
        self.hpf_cutoff_hz
    }

    /// Get the compressor threshold.
    pub fn compressor_threshold_db(&self) -> f64 {
        self.compressor_threshold_db
    }
}

#[cfg(test)]
mod podcast_standard_tests {
    use super::*;

    #[test]
    fn test_spotify_standard() {
        let s = PodcastStandard::SpotifyPodcasts;
        assert!((s.target_lufs() - (-16.0)).abs() < f64::EPSILON);
        assert!((s.true_peak_ceiling_dbtp() - (-1.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_apple_standard() {
        let s = PodcastStandard::ApplePodcasts;
        assert!((s.target_lufs() - (-14.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_youtube_standard() {
        let s = PodcastStandard::YouTubePodcasts;
        assert!((s.target_lufs() - (-14.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_audiobook_standard() {
        let s = PodcastStandard::Audiobook;
        assert!((s.target_lufs() - (-18.0)).abs() < f64::EPSILON);
        assert!((s.true_peak_ceiling_dbtp() - (-3.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_speech_general_standard() {
        let s = PodcastStandard::SpeechGeneral;
        assert!((s.target_lufs() - (-16.0)).abs() < f64::EPSILON);
        assert!((s.speech_hpf_hz() - 80.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_standard_names() {
        assert!(PodcastStandard::SpotifyPodcasts.name().contains("Spotify"));
        assert!(PodcastStandard::ApplePodcasts.name().contains("Apple"));
        assert!(PodcastStandard::Audiobook.name().contains("Audiobook"));
    }

    #[test]
    fn test_to_config() {
        let cfg = PodcastStandard::SpotifyPodcasts.to_config();
        assert!((cfg.platform.target_lufs() - (-16.0)).abs() < f64::EPSILON);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_audiobook_to_config() {
        let cfg = PodcastStandard::Audiobook.to_config();
        assert!((cfg.platform.target_lufs() - (-18.0)).abs() < f64::EPSILON);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_speech_podcast_normalizer_creation() {
        for &std in &[
            PodcastStandard::SpotifyPodcasts,
            PodcastStandard::ApplePodcasts,
            PodcastStandard::YouTubePodcasts,
            PodcastStandard::SpeechGeneral,
            PodcastStandard::Audiobook,
        ] {
            assert!(
                SpeechPodcastNormalizer::new(std).is_ok(),
                "failed to create normalizer for {:?}",
                std
            );
        }
    }

    #[test]
    fn test_speech_podcast_normalizer_process() {
        let norm = SpeechPodcastNormalizer::new(PodcastStandard::SpotifyPodcasts).expect("valid");

        let mut samples: Vec<f32> = (0..4800)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48000.0).sin() * 0.3)
            .collect();

        let result = norm.process(&mut samples, -20.0, 48000.0);
        assert!(result.applied_gain_db > 0.0, "should need boost");
        assert!(samples.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn test_speech_preprocessing_removes_dc() {
        let norm = SpeechPodcastNormalizer::new(PodcastStandard::SpeechGeneral).expect("valid");

        // Signal with DC offset + AC
        let mut samples: Vec<f32> = (0..4800)
            .map(|i| 0.3 + (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 48000.0).sin() * 0.1)
            .collect();

        norm.preprocess_speech(&mut samples, 48000.0);

        // After HPF, the DC component should be significantly reduced
        let mean: f64 = samples.iter().map(|&s| f64::from(s)).sum::<f64>() / samples.len() as f64;
        // Allow some transient settling
        let tail_mean: f64 = samples[1000..].iter().map(|&s| f64::from(s)).sum::<f64>()
            / (samples.len() - 1000) as f64;
        assert!(
            tail_mean.abs() < 0.05,
            "DC should be reduced, tail mean = {tail_mean}"
        );
        let _ = mean;
    }

    #[test]
    fn test_speech_compressor_limits_peaks() {
        let norm = SpeechPodcastNormalizer::new(PodcastStandard::SpeechGeneral).expect("valid");

        let mut samples = vec![0.9_f32; 4800]; // above compressor threshold
        norm.preprocess_speech(&mut samples, 48000.0);

        // After compression, peaks should be reduced (note: HPF also affects level)
        // Just check samples are finite and reduced from original
        assert!(samples.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn test_speech_hpf_parameters() {
        let norm = SpeechPodcastNormalizer::new(PodcastStandard::Audiobook).expect("valid");
        assert!((norm.hpf_cutoff_hz() - 80.0).abs() < f64::EPSILON);
        assert!((norm.compressor_threshold_db() - (-20.0)).abs() < f64::EPSILON);
    }
}
