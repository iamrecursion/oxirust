//! Per-stem loudness management for multi-stem audio mixing.
//!
//! Provides loudness targets and gain balancing for individual audio stems
//! (dialogue, music, effects, ambience, narration) in professional production.

/// Type of audio stem.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum StemType {
    /// Dialogue / speech stem.
    Dialogue,
    /// Music stem.
    Music,
    /// Sound effects stem.
    Effects,
    /// Ambience / atmosphere stem.
    Ambience,
    /// Narration stem.
    Narration,
}

impl StemType {
    /// Get the broadcast target loudness in LUFS for this stem type.
    pub fn target_lufs(&self) -> f64 {
        match self {
            Self::Dialogue => -23.0,
            Self::Music => -30.0,
            Self::Effects => -28.0,
            Self::Ambience => -35.0,
            Self::Narration => -23.0,
        }
    }

    /// Human-readable name for the stem type.
    pub fn name(&self) -> &str {
        match self {
            Self::Dialogue => "Dialogue",
            Self::Music => "Music",
            Self::Effects => "Effects",
            Self::Ambience => "Ambience",
            Self::Narration => "Narration",
        }
    }
}

/// Measured loudness for a single audio stem.
#[derive(Clone, Debug)]
pub struct StemLoudness {
    /// The stem type.
    pub stem_type: StemType,
    /// Measured integrated loudness in LUFS.
    pub measured_lufs: f64,
    /// Measured peak level in dBFS.
    pub peak_dbfs: f64,
}

impl StemLoudness {
    /// Compute the gain in dB needed to reach the stem's default target loudness.
    pub fn required_gain_db(&self) -> f64 {
        self.stem_type.target_lufs() - self.measured_lufs
    }

    /// Check if the stem is within `tolerance_lu` of its target.
    pub fn is_compliant(&self, tolerance_lu: f64) -> bool {
        let target = self.stem_type.target_lufs();
        (self.measured_lufs - target).abs() <= tolerance_lu
    }
}

/// Loudness budget defining relative mix proportions for broadcast.
#[derive(Clone, Debug)]
pub struct LoudnessBudget {
    /// Percentage of mix budget allocated to dialogue (0–100).
    pub dialogue_pct: f32,
    /// Percentage allocated to music.
    pub music_pct: f32,
    /// Percentage allocated to effects.
    pub effects_pct: f32,
    /// Percentage allocated to ambience.
    pub ambience_pct: f32,
}

impl LoudnessBudget {
    /// Standard broadcast budget: D=60%, M=25%, E=10%, A=5%.
    pub fn broadcast() -> Self {
        Self {
            dialogue_pct: 60.0,
            music_pct: 25.0,
            effects_pct: 10.0,
            ambience_pct: 5.0,
        }
    }

    /// Cinematic budget with more music presence: D=50%, M=35%, E=10%, A=5%.
    pub fn cinematic() -> Self {
        Self {
            dialogue_pct: 50.0,
            music_pct: 35.0,
            effects_pct: 10.0,
            ambience_pct: 5.0,
        }
    }

    /// Music-only budget (podcast with music bed): D=70%, M=20%, E=5%, A=5%.
    pub fn podcast() -> Self {
        Self {
            dialogue_pct: 70.0,
            music_pct: 20.0,
            effects_pct: 5.0,
            ambience_pct: 5.0,
        }
    }

    /// Validate that percentages sum to 100 (within tolerance).
    pub fn is_valid(&self) -> bool {
        let total = self.dialogue_pct + self.music_pct + self.effects_pct + self.ambience_pct;
        (total - 100.0).abs() < 0.5
    }
}

/// Stem loudness mixer: computes per-stem gain factors to hit a mix target.
pub struct StemMixer;

impl StemMixer {
    /// Create a new `StemMixer`.
    pub fn new() -> Self {
        Self
    }

    /// Compute gain factors to balance stems to a target mix loudness.
    ///
    /// # Arguments
    /// * `stems` - Slice of (stem type, audio samples). Samples should be mono or
    ///   interleaved; RMS is computed over all provided samples.
    /// * `mix_target_lufs` - Target integrated loudness for the final mix.
    ///
    /// # Returns
    /// A vector of `(stem_name, gain_factor)` pairs, one per input stem.
    pub fn balance(
        &self,
        stems: &[(StemType, Vec<f32>)],
        mix_target_lufs: f64,
    ) -> Vec<(String, f32)> {
        if stems.is_empty() {
            return Vec::new();
        }

        stems
            .iter()
            .map(|(stem_type, samples)| {
                let measured_lufs = measure_rms_lufs(samples);
                let target = stem_type.target_lufs();

                // Primary gain: bring stem to its own target
                let gain_db = if measured_lufs.is_finite() {
                    target - measured_lufs
                } else {
                    0.0
                };

                // Secondary adjustment: offset from mix target relative to dialogue target
                // Dialogue target is anchor; other stems offset from it
                let dialogue_target = StemType::Dialogue.target_lufs();
                let mix_offset = mix_target_lufs - dialogue_target;
                let final_gain_db = gain_db + mix_offset;

                // Clamp to ±30 dB for safety
                let clamped = final_gain_db.clamp(-30.0, 30.0);
                let gain_linear = 10.0_f64.powf(clamped / 20.0) as f32;

                (stem_type.name().to_string(), gain_linear)
            })
            .collect()
    }

    /// Measure per-stem loudness from raw samples.
    pub fn measure_stems(&self, stems: &[(StemType, Vec<f32>)]) -> Vec<StemLoudness> {
        stems
            .iter()
            .map(|(stem_type, samples)| {
                let measured_lufs = measure_rms_lufs(samples);
                let peak_dbfs = measure_peak_dbfs(samples);
                StemLoudness {
                    stem_type: stem_type.clone(),
                    measured_lufs,
                    peak_dbfs,
                }
            })
            .collect()
    }
}

impl Default for StemMixer {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute a simplified LUFS approximation via RMS over all samples.
fn measure_rms_lufs(samples: &[f32]) -> f64 {
    if samples.is_empty() {
        return -f64::INFINITY;
    }
    let mean_sq: f64 = samples
        .iter()
        .map(|&s| (f64::from(s)) * (f64::from(s)))
        .sum::<f64>()
        / samples.len() as f64;

    if mean_sq <= 0.0 {
        return -f64::INFINITY;
    }
    // LUFS ≈ -0.691 + 10 * log10(mean_sq)
    -0.691 + 10.0 * mean_sq.log10()
}

/// Compute peak level in dBFS.
fn measure_peak_dbfs(samples: &[f32]) -> f64 {
    if samples.is_empty() {
        return -f64::INFINITY;
    }
    let peak = samples.iter().map(|&s| s.abs()).fold(0.0_f32, f32::max);
    if peak <= 0.0 {
        -f64::INFINITY
    } else {
        20.0 * f64::from(peak.log10())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stem_type_names() {
        assert_eq!(StemType::Dialogue.name(), "Dialogue");
        assert_eq!(StemType::Music.name(), "Music");
        assert_eq!(StemType::Effects.name(), "Effects");
        assert_eq!(StemType::Ambience.name(), "Ambience");
        assert_eq!(StemType::Narration.name(), "Narration");
    }

    #[test]
    fn test_stem_type_targets() {
        assert!((StemType::Dialogue.target_lufs() - (-23.0)).abs() < 1e-6);
        assert!((StemType::Music.target_lufs() - (-30.0)).abs() < 1e-6);
        assert!((StemType::Effects.target_lufs() - (-28.0)).abs() < 1e-6);
        assert!((StemType::Ambience.target_lufs() - (-35.0)).abs() < 1e-6);
        assert!((StemType::Narration.target_lufs() - (-23.0)).abs() < 1e-6);
    }

    #[test]
    fn test_stem_loudness_required_gain() {
        let sl = StemLoudness {
            stem_type: StemType::Dialogue,
            measured_lufs: -28.0,
            peak_dbfs: -6.0,
        };
        // Target is -23.0, measured is -28.0, so need +5 dB
        assert!((sl.required_gain_db() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_stem_loudness_compliance() {
        let sl = StemLoudness {
            stem_type: StemType::Dialogue,
            measured_lufs: -23.5,
            peak_dbfs: -3.0,
        };
        assert!(sl.is_compliant(1.0)); // Within ±1 LU
        assert!(!sl.is_compliant(0.4)); // Outside ±0.4 LU
    }

    #[test]
    fn test_loudness_budget_broadcast() {
        let b = LoudnessBudget::broadcast();
        assert!((b.dialogue_pct - 60.0).abs() < 1e-6);
        assert!((b.music_pct - 25.0).abs() < 1e-6);
        assert!((b.effects_pct - 10.0).abs() < 1e-6);
        assert!((b.ambience_pct - 5.0).abs() < 1e-6);
        assert!(b.is_valid());
    }

    #[test]
    fn test_loudness_budget_cinematic() {
        let b = LoudnessBudget::cinematic();
        assert!(b.is_valid());
        assert!((b.music_pct - 35.0).abs() < 1e-6);
    }

    #[test]
    fn test_loudness_budget_podcast() {
        let b = LoudnessBudget::podcast();
        assert!(b.is_valid());
        assert!((b.dialogue_pct - 70.0).abs() < 1e-6);
    }

    #[test]
    fn test_stem_mixer_empty() {
        let mixer = StemMixer::new();
        let result = mixer.balance(&[], -23.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_stem_mixer_balance() {
        let mixer = StemMixer::new();
        // Create a simple sine-like signal for each stem
        let sr = 48000;
        let dialogue_samples: Vec<f32> = (0..sr).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();
        let music_samples: Vec<f32> = (0..sr).map(|i| (i as f32 * 0.02).sin() * 0.1).collect();

        let stems = vec![
            (StemType::Dialogue, dialogue_samples),
            (StemType::Music, music_samples),
        ];

        let gains = mixer.balance(&stems, -23.0);
        assert_eq!(gains.len(), 2);

        // Each gain should be a positive finite number
        for (name, gain) in &gains {
            assert!(gain.is_finite(), "Gain for {name} should be finite: {gain}");
            assert!(*gain > 0.0, "Gain for {name} should be positive: {gain}");
        }
    }

    #[test]
    fn test_stem_mixer_measure_stems() {
        let mixer = StemMixer::new();
        let samples: Vec<f32> = (0..4800).map(|i| (i as f32 * 0.01).sin() * 0.3).collect();
        let stems = vec![(StemType::Dialogue, samples)];
        let measurements = mixer.measure_stems(&stems);
        assert_eq!(measurements.len(), 1);
        assert!(measurements[0].measured_lufs.is_finite());
        assert!(measurements[0].peak_dbfs.is_finite());
    }

    #[test]
    fn test_measure_rms_lufs_empty() {
        let lufs = measure_rms_lufs(&[]);
        assert!(!lufs.is_finite());
    }

    #[test]
    fn test_measure_rms_lufs_silence() {
        let samples = vec![0.0f32; 1000];
        let lufs = measure_rms_lufs(&samples);
        assert!(!lufs.is_finite() || lufs < -100.0);
    }

    #[test]
    fn test_measure_peak_dbfs() {
        let samples = vec![0.0f32, 0.5, -0.8, 0.3];
        let peak = measure_peak_dbfs(&samples);
        let expected = 20.0 * 0.8_f64.log10();
        assert!((peak - expected).abs() < 0.01);
    }
}
