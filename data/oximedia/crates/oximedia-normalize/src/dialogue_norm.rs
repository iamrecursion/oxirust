#![allow(dead_code)]
//! Dialogue normalization — ensure spoken-word content hits a target integrated loudness.

/// Measured loudness properties of a dialogue track.
#[derive(Debug, Clone)]
pub struct DialogueLoudness {
    /// Integrated loudness in LKFS/LUFS.
    pub integrated_lkfs: f64,
    /// Loudness range (LRA) in LU.
    pub loudness_range_lu: f64,
    /// True peak in dBTP.
    pub true_peak_dbtp: f64,
    /// Proportion of the program that is speech (0.0–1.0).
    pub speech_ratio: f64,
}

impl DialogueLoudness {
    /// Create a new measurement.
    pub fn new(
        integrated_lkfs: f64,
        loudness_range_lu: f64,
        true_peak_dbtp: f64,
        speech_ratio: f64,
    ) -> Self {
        Self {
            integrated_lkfs,
            loudness_range_lu,
            true_peak_dbtp,
            speech_ratio: speech_ratio.clamp(0.0, 1.0),
        }
    }

    /// Whether the measured loudness is within the given spec's tolerance.
    pub fn is_within_spec(&self, config: &DialogueNormConfig) -> bool {
        let diff = (self.integrated_lkfs - config.target_lkfs).abs();
        diff <= config.tolerance_lu
    }

    /// How many dB of gain correction are needed to reach target.
    pub fn correction_needed_db(&self, target_lkfs: f64) -> f64 {
        target_lkfs - self.integrated_lkfs
    }
}

/// Configuration for dialogue normalization.
#[derive(Debug, Clone)]
pub struct DialogueNormConfig {
    /// Target integrated loudness (LKFS).
    pub target_lkfs: f64,
    /// Acceptable deviation before correction is applied (LU).
    pub tolerance_lu: f64,
    /// Maximum gain allowed in dB.
    pub max_gain_db: f64,
    /// Maximum attenuation allowed in dB (positive value).
    pub max_attenuation_db: f64,
    /// True peak ceiling in dBTP.
    pub true_peak_ceiling_dbtp: f64,
}

impl DialogueNormConfig {
    /// ATSC A/85 dialogue normalization: −24 LKFS ±2 LU.
    pub fn atsc() -> Self {
        Self {
            target_lkfs: -24.0,
            tolerance_lu: 2.0,
            max_gain_db: 15.0,
            max_attenuation_db: 20.0,
            true_peak_ceiling_dbtp: -2.0,
        }
    }

    /// EBU R128 dialogue normalization: −23 LUFS ±1 LU.
    pub fn ebu_r128() -> Self {
        Self {
            target_lkfs: -23.0,
            tolerance_lu: 1.0,
            max_gain_db: 15.0,
            max_attenuation_db: 20.0,
            true_peak_ceiling_dbtp: -1.0,
        }
    }

    /// Custom configuration.
    pub fn custom(target_lkfs: f64, tolerance_lu: f64) -> Self {
        Self {
            target_lkfs,
            tolerance_lu,
            max_gain_db: 20.0,
            max_attenuation_db: 20.0,
            true_peak_ceiling_dbtp: -1.0,
        }
    }

    /// Return the target LKFS.
    pub fn target_lkfs(&self) -> f64 {
        self.target_lkfs
    }
}

impl Default for DialogueNormConfig {
    fn default() -> Self {
        Self::atsc()
    }
}

/// Result of a dialogue normalization operation.
#[derive(Debug, Clone)]
pub struct DialogueNormResult {
    /// Gain correction applied (positive = boost, negative = cut).
    pub applied_gain_db: f64,
    /// Measured loudness before correction.
    pub input_loudness: DialogueLoudness,
    /// Estimated loudness after correction.
    pub output_lkfs: f64,
    /// Whether the true peak ceiling was hit, requiring a limiter pass.
    pub limiter_engaged: bool,
    /// Whether the final result is within spec.
    pub within_spec: bool,
}

impl DialogueNormResult {
    /// The dB correction applied to reach the target.
    pub fn correction_db(&self) -> f64 {
        self.applied_gain_db
    }

    /// Whether the correction was a boost (positive gain).
    pub fn was_boost(&self) -> bool {
        self.applied_gain_db > 0.0
    }

    /// Whether the correction was an attenuation.
    pub fn was_attenuation(&self) -> bool {
        self.applied_gain_db < 0.0
    }
}

/// Dialogue normalizer: analyse loudness then apply gain correction.
#[derive(Debug, Default)]
pub struct DialogueNormalizer {
    config: DialogueNormConfig,
}

impl DialogueNormalizer {
    /// Create a new normalizer with the given config.
    pub fn new(config: DialogueNormConfig) -> Self {
        Self { config }
    }

    /// Analyse a pre-measured `DialogueLoudness` against the configured target.
    pub fn analyze(&self, loudness: &DialogueLoudness) -> f64 {
        loudness.correction_needed_db(self.config.target_lkfs)
    }

    /// Apply normalization: compute the correction, clamp to limits, and return a result.
    pub fn apply(&self, loudness: DialogueLoudness) -> DialogueNormResult {
        let raw_correction = self.analyze(&loudness);

        // Clamp correction to configured limits.
        let applied_gain_db = if raw_correction > 0.0 {
            raw_correction.min(self.config.max_gain_db)
        } else {
            raw_correction.max(-self.config.max_attenuation_db)
        };

        let output_lkfs = loudness.integrated_lkfs + applied_gain_db;
        let output_true_peak = loudness.true_peak_dbtp + applied_gain_db;
        let limiter_engaged = output_true_peak > self.config.true_peak_ceiling_dbtp;

        let output_loudness_for_spec = DialogueLoudness::new(
            output_lkfs,
            loudness.loudness_range_lu,
            output_true_peak,
            loudness.speech_ratio,
        );
        let within_spec = output_loudness_for_spec.is_within_spec(&self.config);

        DialogueNormResult {
            applied_gain_db,
            input_loudness: loudness,
            output_lkfs,
            limiter_engaged,
            within_spec,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_loudness(lkfs: f64) -> DialogueLoudness {
        DialogueLoudness::new(lkfs, 8.0, -6.0, 0.7)
    }

    #[test]
    fn test_is_within_spec_true() {
        let cfg = DialogueNormConfig::atsc();
        let loud = sample_loudness(-24.5); // within ±2 LU of -24
        assert!(loud.is_within_spec(&cfg));
    }

    #[test]
    fn test_is_within_spec_false() {
        let cfg = DialogueNormConfig::atsc();
        let loud = sample_loudness(-30.0); // 6 LU below target
        assert!(!loud.is_within_spec(&cfg));
    }

    #[test]
    fn test_correction_needed_db_boost() {
        let loud = sample_loudness(-28.0);
        let corr = loud.correction_needed_db(-24.0);
        assert!((corr - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_correction_needed_db_cut() {
        let loud = sample_loudness(-18.0);
        let corr = loud.correction_needed_db(-23.0);
        assert!((corr - (-5.0)).abs() < 1e-9);
    }

    #[test]
    fn test_config_target_lkfs_atsc() {
        let cfg = DialogueNormConfig::atsc();
        assert!((cfg.target_lkfs() - (-24.0)).abs() < 1e-9);
    }

    #[test]
    fn test_config_target_lkfs_ebu() {
        let cfg = DialogueNormConfig::ebu_r128();
        assert!((cfg.target_lkfs() - (-23.0)).abs() < 1e-9);
    }

    #[test]
    fn test_config_custom() {
        let cfg = DialogueNormConfig::custom(-20.0, 0.5);
        assert!((cfg.target_lkfs - (-20.0)).abs() < 1e-9);
        assert!((cfg.tolerance_lu - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_apply_boost() {
        let norm = DialogueNormalizer::new(DialogueNormConfig::atsc());
        let result = norm.apply(sample_loudness(-30.0));
        assert!(result.applied_gain_db > 0.0);
        assert!(result.was_boost());
        assert!(!result.was_attenuation());
    }

    #[test]
    fn test_apply_attenuation() {
        let norm = DialogueNormalizer::new(DialogueNormConfig::atsc());
        let result = norm.apply(sample_loudness(-10.0));
        assert!(result.applied_gain_db < 0.0);
        assert!(result.was_attenuation());
    }

    #[test]
    fn test_apply_clamped_boost() {
        let mut cfg = DialogueNormConfig::atsc();
        cfg.max_gain_db = 5.0;
        let norm = DialogueNormalizer::new(cfg);
        let result = norm.apply(sample_loudness(-40.0)); // needs +16 dB
        assert!((result.applied_gain_db - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_apply_correction_db_accessor() {
        let norm = DialogueNormalizer::new(DialogueNormConfig::ebu_r128());
        let result = norm.apply(sample_loudness(-30.0));
        assert!((result.correction_db() - result.applied_gain_db).abs() < 1e-9);
    }

    #[test]
    fn test_output_lkfs_after_apply() {
        let norm = DialogueNormalizer::new(DialogueNormConfig::atsc());
        let result = norm.apply(sample_loudness(-28.0));
        let expected = -28.0 + result.applied_gain_db;
        assert!((result.output_lkfs - expected).abs() < 1e-9);
    }

    #[test]
    fn test_speech_ratio_clamped() {
        let loud = DialogueLoudness::new(-23.0, 6.0, -2.0, 1.5);
        assert!((loud.speech_ratio - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_within_spec_after_correction() {
        let norm = DialogueNormalizer::new(DialogueNormConfig::atsc());
        let result = norm.apply(sample_loudness(-24.0));
        assert!(result.within_spec);
    }
}
