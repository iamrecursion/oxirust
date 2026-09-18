//! Cinema loudness normalization standards.
//!
//! Implements loudness targets for cinema / theatrical distribution:
//!
//! - **Dolby Atmos** (-27 LUFS dialogue-gated)
//! - **DCI (Digital Cinema Initiative)** (-24 LEQA, Leq(A) measurement)
//! - **Dolby Theatrical** (-31 LUFS dialogue-gated, older spec)
//!
//! # Algorithm
//!
//! 1. Measure integrated loudness using dialogue-gated measurement:
//!    only regions where dialogue is present contribute to the measurement.
//! 2. Compute gain delta: `gain = target_lufs - measured_lufs`.
//! 3. Clamp gain to configured limits.
//! 4. Apply gain to all samples.
//! 5. Apply true peak limiter to prevent inter-sample clipping.
//!
//! The dialogue-gated approach differs from EBU R128's absolute gate:
//! cinema content has long passages of music/effects that should not
//! skew the dialogue-anchored loudness measurement.

/// Cinema loudness standard preset.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CinemaStandard {
    /// Dolby Atmos theatrical: -27 LUFS dialogue-gated, -1 dBTP max.
    DolbyAtmos,
    /// DCI standard: -24 LEQA (A-weighted Leq).
    Dci,
    /// Legacy Dolby theatrical: -31 LUFS dialogue-gated.
    DolbyLegacy,
    /// Custom cinema target.
    Custom {
        /// Target loudness in LUFS (dialogue-gated).
        target_lufs: f64,
        /// Maximum true peak in dBTP.
        true_peak_ceiling_dbtp: f64,
    },
}

impl CinemaStandard {
    /// Target loudness in LUFS (dialogue-gated).
    pub fn target_lufs(self) -> f64 {
        match self {
            Self::DolbyAtmos => -27.0,
            Self::Dci => -24.0,
            Self::DolbyLegacy => -31.0,
            Self::Custom { target_lufs, .. } => target_lufs,
        }
    }

    /// Maximum true peak level in dBTP.
    pub fn true_peak_ceiling_dbtp(self) -> f64 {
        match self {
            Self::DolbyAtmos => -1.0,
            Self::Dci => -3.0,
            Self::DolbyLegacy => -3.0,
            Self::Custom {
                true_peak_ceiling_dbtp,
                ..
            } => true_peak_ceiling_dbtp,
        }
    }

    /// Human-readable name.
    pub fn name(self) -> &'static str {
        match self {
            Self::DolbyAtmos => "Dolby Atmos Theatrical",
            Self::Dci => "DCI Cinema",
            Self::DolbyLegacy => "Dolby Legacy Theatrical",
            Self::Custom { .. } => "Custom Cinema",
        }
    }
}

/// Configuration for cinema loudness normalization.
#[derive(Debug, Clone)]
pub struct CinemaLoudnessConfig {
    /// Cinema standard to target.
    pub standard: CinemaStandard,
    /// Maximum allowed gain boost in dB.
    pub max_gain_db: f64,
    /// Maximum allowed attenuation in dB (positive value).
    pub max_attenuation_db: f64,
    /// Enable brick-wall peak limiter after gain application.
    pub enable_limiter: bool,
    /// Tolerance in LU: if measured loudness is within this range of target,
    /// no normalization is applied.
    pub tolerance_lu: f64,
    /// Dialogue gate threshold in dBFS: frames below this are excluded from
    /// dialogue-gated measurement.
    pub dialogue_gate_dbfs: f64,
}

impl CinemaLoudnessConfig {
    /// Create a configuration for the given cinema standard with sensible defaults.
    pub fn new(standard: CinemaStandard) -> Self {
        Self {
            standard,
            max_gain_db: 15.0,
            max_attenuation_db: 30.0,
            enable_limiter: true,
            tolerance_lu: 1.0,
            dialogue_gate_dbfs: -50.0,
        }
    }

    /// Create a Dolby Atmos configuration (-27 LUFS dialogue-gated).
    pub fn dolby_atmos() -> Self {
        Self::new(CinemaStandard::DolbyAtmos)
    }

    /// Create a DCI cinema configuration (-24 LEQA).
    pub fn dci() -> Self {
        Self::new(CinemaStandard::Dci)
    }

    /// Create a legacy Dolby theatrical configuration (-31 LUFS).
    pub fn dolby_legacy() -> Self {
        Self::new(CinemaStandard::DolbyLegacy)
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
        if self.dialogue_gate_dbfs > 0.0 || self.dialogue_gate_dbfs < -120.0 {
            return Err(format!(
                "dialogue_gate_dbfs must be in [-120, 0] dBFS, got {}",
                self.dialogue_gate_dbfs
            ));
        }
        Ok(())
    }
}

/// Result of cinema loudness normalization.
#[derive(Debug, Clone)]
pub struct CinemaLoudnessResult {
    /// Cinema standard used.
    pub standard: CinemaStandard,
    /// Measured dialogue-gated loudness in LUFS.
    pub measured_lufs: f64,
    /// Target loudness in LUFS.
    pub target_lufs: f64,
    /// Gain applied in dB.
    pub applied_gain_db: f64,
    /// Whether the limiter was engaged.
    pub limiter_engaged: bool,
    /// Estimated output loudness after normalization (LUFS).
    pub output_lufs: f64,
    /// Number of frames included in dialogue-gated measurement.
    pub dialogue_frames: usize,
    /// Total frames analyzed.
    pub total_frames: usize,
}

impl CinemaLoudnessResult {
    /// True if normalization was a no-op (already within tolerance).
    pub fn was_noop(&self) -> bool {
        self.applied_gain_db.abs() < 1e-6
    }

    /// Ratio of dialogue frames to total frames.
    pub fn dialogue_ratio(&self) -> f64 {
        if self.total_frames == 0 {
            0.0
        } else {
            self.dialogue_frames as f64 / self.total_frames as f64
        }
    }
}

/// Cinema loudness normalizer with dialogue-gated measurement.
///
/// The normalizer computes dialogue-gated integrated loudness:
/// only frames where the RMS level exceeds the dialogue gate threshold
/// contribute to the loudness calculation. This ensures that quiet
/// ambient passages, music, and effects don't dilute the dialogue measurement.
pub struct CinemaLoudnessNormalizer {
    config: CinemaLoudnessConfig,
}

impl CinemaLoudnessNormalizer {
    /// Create a new cinema loudness normalizer.
    pub fn new(config: CinemaLoudnessConfig) -> Result<Self, String> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Measure dialogue-gated loudness from interleaved f32 samples.
    ///
    /// Breaks audio into frames of `frame_size` samples per channel,
    /// measures the RMS of each frame, and includes only frames above
    /// the dialogue gate in the integrated loudness calculation.
    ///
    /// Returns `(dialogue_gated_lufs, dialogue_frames, total_frames)`.
    pub fn measure_dialogue_gated(
        &self,
        samples: &[f32],
        channels: usize,
        frame_size: usize,
    ) -> (f64, usize, usize) {
        if samples.is_empty() || channels == 0 || frame_size == 0 {
            return (-100.0, 0, 0);
        }

        let samples_per_frame = frame_size * channels;
        let gate_linear_sq = {
            let dbfs = self.config.dialogue_gate_dbfs;
            let linear = 10.0_f64.powf(dbfs / 20.0);
            linear * linear
        };

        let mut sum_sq_gated = 0.0_f64;
        let mut gated_sample_count = 0usize;
        let mut dialogue_frames = 0usize;
        let mut total_frames = 0usize;

        let mut pos = 0usize;
        while pos + samples_per_frame <= samples.len() {
            let frame = &samples[pos..pos + samples_per_frame];
            let frame_rms_sq: f64 = frame
                .iter()
                .map(|&s| f64::from(s) * f64::from(s))
                .sum::<f64>()
                / frame.len() as f64;

            total_frames += 1;

            if frame_rms_sq > gate_linear_sq {
                // This frame counts as dialogue
                dialogue_frames += 1;
                sum_sq_gated += frame
                    .iter()
                    .map(|&s| f64::from(s) * f64::from(s))
                    .sum::<f64>();
                gated_sample_count += frame.len();
            }

            pos += samples_per_frame;
        }

        let gated_lufs = if gated_sample_count > 0 {
            let mean_sq = sum_sq_gated / gated_sample_count as f64;
            // LUFS ≈ -0.691 + 10*log10(mean_sq)  (simplified K-weighted model)
            -0.691 + 10.0 * mean_sq.max(1e-20).log10()
        } else {
            -100.0
        };

        (gated_lufs, dialogue_frames, total_frames)
    }

    /// Compute the required gain from a pre-measured dialogue-gated loudness.
    pub fn compute_gain(&self, measured_lufs: f64) -> f64 {
        let target = self.config.standard.target_lufs();
        let raw_gain = target - measured_lufs;

        if raw_gain.abs() <= self.config.tolerance_lu {
            0.0
        } else if raw_gain > 0.0 {
            raw_gain.min(self.config.max_gain_db)
        } else {
            raw_gain.max(-self.config.max_attenuation_db)
        }
    }

    /// Full processing pipeline: measure, compute gain, apply, limit.
    ///
    /// `frame_size` is the number of samples per channel per analysis frame.
    pub fn process(
        &self,
        samples: &mut [f32],
        channels: usize,
        frame_size: usize,
    ) -> CinemaLoudnessResult {
        let (measured_lufs, dialogue_frames, total_frames) =
            self.measure_dialogue_gated(samples, channels, frame_size);

        let applied_gain_db = self.compute_gain(measured_lufs);

        // Apply gain
        if applied_gain_db.abs() > 1e-9 {
            let gain = 10.0_f32.powf(applied_gain_db as f32 / 20.0);
            for s in samples.iter_mut() {
                *s *= gain;
            }
        }

        // Brick-wall peak limiter
        let mut limiter_engaged = false;
        if self.config.enable_limiter {
            let ceiling =
                10.0_f32.powf(self.config.standard.true_peak_ceiling_dbtp() as f32 / 20.0);
            for s in samples.iter_mut() {
                if s.abs() > ceiling {
                    *s = s.signum() * ceiling;
                    limiter_engaged = true;
                }
            }
        }

        CinemaLoudnessResult {
            standard: self.config.standard,
            measured_lufs,
            target_lufs: self.config.standard.target_lufs(),
            applied_gain_db,
            limiter_engaged,
            output_lufs: measured_lufs + applied_gain_db,
            dialogue_frames,
            total_frames,
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &CinemaLoudnessConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── CinemaStandard ────────────────────────────────────────────────────

    #[test]
    fn test_dolby_atmos_target() {
        assert!((CinemaStandard::DolbyAtmos.target_lufs() - (-27.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_dci_target() {
        assert!((CinemaStandard::Dci.target_lufs() - (-24.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_dolby_legacy_target() {
        assert!((CinemaStandard::DolbyLegacy.target_lufs() - (-31.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_custom_target() {
        let s = CinemaStandard::Custom {
            target_lufs: -25.0,
            true_peak_ceiling_dbtp: -2.0,
        };
        assert!((s.target_lufs() - (-25.0)).abs() < f64::EPSILON);
        assert!((s.true_peak_ceiling_dbtp() - (-2.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_true_peak_ceilings() {
        assert!(
            (CinemaStandard::DolbyAtmos.true_peak_ceiling_dbtp() - (-1.0)).abs() < f64::EPSILON
        );
        assert!((CinemaStandard::Dci.true_peak_ceiling_dbtp() - (-3.0)).abs() < f64::EPSILON);
        assert!(
            (CinemaStandard::DolbyLegacy.true_peak_ceiling_dbtp() - (-3.0)).abs() < f64::EPSILON
        );
    }

    #[test]
    fn test_standard_names() {
        assert_eq!(CinemaStandard::DolbyAtmos.name(), "Dolby Atmos Theatrical");
        assert_eq!(CinemaStandard::Dci.name(), "DCI Cinema");
        assert_eq!(
            CinemaStandard::DolbyLegacy.name(),
            "Dolby Legacy Theatrical"
        );
    }

    // ─── CinemaLoudnessConfig ──────────────────────────────────────────────

    #[test]
    fn test_config_dolby_atmos() {
        let cfg = CinemaLoudnessConfig::dolby_atmos();
        assert!((cfg.standard.target_lufs() - (-27.0)).abs() < f64::EPSILON);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_dci() {
        let cfg = CinemaLoudnessConfig::dci();
        assert!((cfg.standard.target_lufs() - (-24.0)).abs() < f64::EPSILON);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_dolby_legacy() {
        let cfg = CinemaLoudnessConfig::dolby_legacy();
        assert!((cfg.standard.target_lufs() - (-31.0)).abs() < f64::EPSILON);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_validation_invalid_gain() {
        let mut cfg = CinemaLoudnessConfig::dolby_atmos();
        cfg.max_gain_db = -1.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validation_invalid_tolerance() {
        let mut cfg = CinemaLoudnessConfig::dolby_atmos();
        cfg.tolerance_lu = 15.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validation_invalid_gate() {
        let mut cfg = CinemaLoudnessConfig::dolby_atmos();
        cfg.dialogue_gate_dbfs = 5.0;
        assert!(cfg.validate().is_err());
    }

    // ─── CinemaLoudnessNormalizer ──────────────────────────────────────────

    #[test]
    fn test_normalizer_creation() {
        let cfg = CinemaLoudnessConfig::dolby_atmos();
        assert!(CinemaLoudnessNormalizer::new(cfg).is_ok());
    }

    #[test]
    fn test_compute_gain_needs_boost() {
        let norm =
            CinemaLoudnessNormalizer::new(CinemaLoudnessConfig::dolby_atmos()).expect("valid");
        let gain = norm.compute_gain(-30.0); // needs +3 dB to reach -27
        assert!((gain - 3.0).abs() < 0.01, "expected +3 dB, got {gain}");
    }

    #[test]
    fn test_compute_gain_needs_attenuation() {
        let norm =
            CinemaLoudnessNormalizer::new(CinemaLoudnessConfig::dolby_atmos()).expect("valid");
        let gain = norm.compute_gain(-22.0); // needs -5 dB to reach -27
        assert!((gain - (-5.0)).abs() < 0.01, "expected -5 dB, got {gain}");
    }

    #[test]
    fn test_compute_gain_within_tolerance() {
        let norm =
            CinemaLoudnessNormalizer::new(CinemaLoudnessConfig::dolby_atmos()).expect("valid");
        let gain = norm.compute_gain(-27.5); // within ±1.0 LU tolerance
        assert!(gain.abs() < 1e-6, "expected noop, got gain = {gain}");
    }

    #[test]
    fn test_compute_gain_clamped() {
        let mut cfg = CinemaLoudnessConfig::dolby_atmos();
        cfg.max_gain_db = 5.0;
        let norm = CinemaLoudnessNormalizer::new(cfg).expect("valid");
        let gain = norm.compute_gain(-40.0); // needs +13 dB but clamped to +5
        assert!((gain - 5.0).abs() < 0.01, "expected 5 dB, got {gain}");
    }

    #[test]
    fn test_measure_dialogue_gated_empty() {
        let norm =
            CinemaLoudnessNormalizer::new(CinemaLoudnessConfig::dolby_atmos()).expect("valid");
        let (lufs, df, tf) = norm.measure_dialogue_gated(&[], 2, 480);
        assert!(lufs < -90.0);
        assert_eq!(df, 0);
        assert_eq!(tf, 0);
    }

    #[test]
    fn test_measure_dialogue_gated_with_signal() {
        let norm =
            CinemaLoudnessNormalizer::new(CinemaLoudnessConfig::dolby_atmos()).expect("valid");

        // Create a signal loud enough to pass the gate
        let samples: Vec<f32> = (0..4800)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48000.0).sin() * 0.3)
            .collect();

        let (lufs, df, tf) = norm.measure_dialogue_gated(&samples, 1, 480);
        assert!(lufs > -100.0, "expected valid measurement, got {lufs}");
        assert!(df > 0, "expected some dialogue frames");
        assert!(tf > 0, "expected some total frames");
    }

    #[test]
    fn test_measure_dialogue_gated_silence_excluded() {
        let mut cfg = CinemaLoudnessConfig::dolby_atmos();
        cfg.dialogue_gate_dbfs = -40.0; // strict gate
        let norm = CinemaLoudnessNormalizer::new(cfg).expect("valid");

        // Very quiet signal: should be gated out
        let samples = vec![0.0001_f32; 4800];
        let (_, df, tf) = norm.measure_dialogue_gated(&samples, 1, 480);
        assert_eq!(df, 0, "silence should be gated out");
        assert!(tf > 0);
    }

    #[test]
    fn test_process_applies_gain() {
        let norm =
            CinemaLoudnessNormalizer::new(CinemaLoudnessConfig::dolby_atmos()).expect("valid");

        let mut samples: Vec<f32> = (0..4800)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48000.0).sin() * 0.3)
            .collect();
        let original_rms: f64 = samples
            .iter()
            .map(|&s| f64::from(s) * f64::from(s))
            .sum::<f64>()
            / samples.len() as f64;

        let result = norm.process(&mut samples, 1, 480);

        let output_rms: f64 = samples
            .iter()
            .map(|&s| f64::from(s) * f64::from(s))
            .sum::<f64>()
            / samples.len() as f64;

        // If gain was applied, RMS should differ
        if result.applied_gain_db.abs() > 0.1 {
            assert!(
                (output_rms - original_rms).abs() > 1e-6,
                "expected RMS change after gain"
            );
        }
        assert!(samples.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn test_process_limiter_engages() {
        let cfg = CinemaLoudnessConfig::new(CinemaStandard::Custom {
            target_lufs: -10.0, // very loud target
            true_peak_ceiling_dbtp: -3.0,
        });
        let norm = CinemaLoudnessNormalizer::new(cfg).expect("valid");

        let mut samples = vec![0.8_f32; 4800]; // will clip after boost
        let result = norm.process(&mut samples, 1, 480);

        // Check ceiling is respected
        let ceiling = 10.0_f32.powf(-3.0 / 20.0);
        for &s in &samples {
            assert!(
                s.abs() <= ceiling + 1e-5,
                "sample {} exceeds ceiling {}",
                s,
                ceiling
            );
        }
        // If boost was large enough, limiter should engage
        if result.applied_gain_db > 1.0 {
            assert!(result.limiter_engaged);
        }
    }

    #[test]
    fn test_cinema_result_dialogue_ratio() {
        let result = CinemaLoudnessResult {
            standard: CinemaStandard::DolbyAtmos,
            measured_lufs: -27.0,
            target_lufs: -27.0,
            applied_gain_db: 0.0,
            limiter_engaged: false,
            output_lufs: -27.0,
            dialogue_frames: 8,
            total_frames: 10,
        };
        assert!((result.dialogue_ratio() - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_cinema_result_was_noop() {
        let result = CinemaLoudnessResult {
            standard: CinemaStandard::DolbyAtmos,
            measured_lufs: -27.0,
            target_lufs: -27.0,
            applied_gain_db: 0.0,
            limiter_engaged: false,
            output_lufs: -27.0,
            dialogue_frames: 10,
            total_frames: 10,
        };
        assert!(result.was_noop());
    }

    #[test]
    fn test_cinema_result_dialogue_ratio_zero_frames() {
        let result = CinemaLoudnessResult {
            standard: CinemaStandard::DolbyAtmos,
            measured_lufs: -100.0,
            target_lufs: -27.0,
            applied_gain_db: 0.0,
            limiter_engaged: false,
            output_lufs: -100.0,
            dialogue_frames: 0,
            total_frames: 0,
        };
        assert!((result.dialogue_ratio()).abs() < f64::EPSILON);
    }
}
