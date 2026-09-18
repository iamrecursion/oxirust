#![allow(dead_code)]
//! Transient repair and reconstruction for degraded audio.
//!
//! This module detects and repairs damaged transients in audio recordings,
//! including missing attacks, flattened transient peaks, and transient
//! smearing caused by analog degradation or lossy compression artifacts.

/// Strategy used to repair detected transient damage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepairStrategy {
    /// Reshape the transient envelope to restore attack characteristics.
    EnvelopeReshape,
    /// Reconstruct the transient from surrounding context using interpolation.
    ContextInterpolation,
    /// Boost the existing transient energy to compensate for loss.
    EnergyBoost,
    /// Replace the damaged transient with a synthetic model.
    SyntheticReplace,
}

/// Configuration for the transient repair processor.
#[derive(Debug, Clone)]
pub struct TransientRepairConfig {
    /// Detection threshold for identifying damaged transients (0.0..1.0).
    pub detection_threshold: f32,
    /// Attack time in milliseconds for transient envelope analysis.
    pub attack_ms: f32,
    /// Release time in milliseconds for transient envelope analysis.
    pub release_ms: f32,
    /// Maximum boost in dB when using `EnergyBoost` strategy.
    pub max_boost_db: f32,
    /// Strategy to use for repair.
    pub strategy: RepairStrategy,
    /// Window size in samples for transient analysis.
    pub window_size: usize,
    /// Look-ahead in samples to detect upcoming transients.
    pub lookahead_samples: usize,
}

impl Default for TransientRepairConfig {
    fn default() -> Self {
        Self {
            detection_threshold: 0.3,
            attack_ms: 5.0,
            release_ms: 50.0,
            max_boost_db: 12.0,
            strategy: RepairStrategy::EnvelopeReshape,
            window_size: 512,
            lookahead_samples: 64,
        }
    }
}

/// A detected region where a transient appears damaged.
#[derive(Debug, Clone)]
pub struct DamagedTransient {
    /// Start sample index of the damaged region.
    pub start: usize,
    /// End sample index of the damaged region.
    pub end: usize,
    /// Severity of the damage (0.0 = minor, 1.0 = severe).
    pub severity: f32,
    /// Expected peak level based on context.
    pub expected_peak: f32,
    /// Actual measured peak level.
    pub actual_peak: f32,
}

/// Transient repair processor that detects and fixes degraded transients.
#[derive(Debug, Clone)]
pub struct TransientRepairer {
    /// Configuration for repair operations.
    config: TransientRepairConfig,
    /// Envelope follower state (attack coefficient).
    attack_coeff: f32,
    /// Envelope follower state (release coefficient).
    release_coeff: f32,
}

impl TransientRepairer {
    /// Create a new transient repairer with the given configuration and sample rate.
    #[allow(clippy::cast_precision_loss)]
    pub fn new(config: TransientRepairConfig, sample_rate: u32) -> Self {
        let sr = sample_rate as f64;
        let attack_coeff = ((-1.0_f64) / (config.attack_ms as f64 * 0.001 * sr)).exp() as f32;
        let release_coeff = ((-1.0_f64) / (config.release_ms as f64 * 0.001 * sr)).exp() as f32;
        Self {
            config,
            attack_coeff,
            release_coeff,
        }
    }

    /// Create a new transient repairer with default configuration.
    pub fn with_defaults(sample_rate: u32) -> Self {
        Self::new(TransientRepairConfig::default(), sample_rate)
    }

    /// Compute the envelope of the signal using attack/release follower.
    fn compute_envelope(&self, samples: &[f32]) -> Vec<f32> {
        let mut envelope = vec![0.0_f32; samples.len()];
        let mut current = 0.0_f32;
        for (i, &s) in samples.iter().enumerate() {
            let abs_s = s.abs();
            if abs_s > current {
                current = self.attack_coeff * current + (1.0 - self.attack_coeff) * abs_s;
            } else {
                current = self.release_coeff * current + (1.0 - self.release_coeff) * abs_s;
            }
            envelope[i] = current;
        }
        envelope
    }

    /// Detect damaged transients in the audio signal.
    pub fn detect(&self, samples: &[f32]) -> Vec<DamagedTransient> {
        let envelope = self.compute_envelope(samples);
        let mut damaged = Vec::new();
        let win = self.config.window_size.max(4);

        let mut i = win;
        while i + win < samples.len() {
            // Compute local energy before and at current window
            let pre_energy: f32 = envelope[i.saturating_sub(win)..i]
                .iter()
                .map(|v| v * v)
                .sum::<f32>()
                / win as f32;
            let cur_energy: f32 =
                envelope[i..i + win].iter().map(|v| v * v).sum::<f32>() / win as f32;

            let pre_rms = pre_energy.sqrt();
            let cur_rms = cur_energy.sqrt();

            // A transient is expected where there is a sudden rise in envelope
            // Damage is detected when the rise is muted compared to context
            if pre_rms > self.config.detection_threshold && cur_rms < pre_rms * 0.5 {
                let expected = pre_rms;
                let actual = cur_rms;
                let severity = if expected > 0.0 {
                    (1.0 - actual / expected).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                damaged.push(DamagedTransient {
                    start: i,
                    end: (i + win).min(samples.len()),
                    severity,
                    expected_peak: expected,
                    actual_peak: actual,
                });
                i += win; // skip past this region
            } else {
                i += win / 2;
            }
        }
        damaged
    }

    /// Repair damaged transients in the audio signal.
    pub fn repair(&self, samples: &[f32]) -> Vec<f32> {
        let damaged_regions = self.detect(samples);
        let mut output = samples.to_vec();

        for region in &damaged_regions {
            match self.config.strategy {
                RepairStrategy::EnvelopeReshape => {
                    self.reshape_envelope(&mut output, region);
                }
                RepairStrategy::ContextInterpolation => {
                    self.interpolate_context(&mut output, region);
                }
                RepairStrategy::EnergyBoost => {
                    self.boost_energy(&mut output, region);
                }
                RepairStrategy::SyntheticReplace => {
                    self.synthetic_replace(&mut output, region);
                }
            }
        }
        output
    }

    /// Reshape the envelope of the damaged region to restore attack.
    fn reshape_envelope(&self, output: &mut [f32], region: &DamagedTransient) {
        let len = region.end - region.start;
        if len == 0 || region.expected_peak <= 0.0 {
            return;
        }
        for i in 0..len {
            let idx = region.start + i;
            if idx >= output.len() {
                break;
            }
            // Apply an attack-shaped gain curve
            #[allow(clippy::cast_precision_loss)]
            let t = i as f32 / len as f32;
            let gain = 1.0 + region.severity * (1.0 - t);
            output[idx] = (output[idx] * gain).clamp(-1.0, 1.0);
        }
    }

    /// Interpolate the damaged region from surrounding context.
    fn interpolate_context(&self, output: &mut [f32], region: &DamagedTransient) {
        let start = region.start;
        let end = region.end.min(output.len());
        if start == 0 || end >= output.len() {
            return;
        }
        let val_before = output[start.saturating_sub(1)];
        let val_after = if end < output.len() {
            output[end]
        } else {
            val_before
        };
        let len = end - start;
        for i in 0..len {
            #[allow(clippy::cast_precision_loss)]
            let t = (i + 1) as f32 / (len + 1) as f32;
            output[start + i] = val_before + (val_after - val_before) * t;
        }
    }

    /// Boost the energy of the damaged region.
    fn boost_energy(&self, output: &mut [f32], region: &DamagedTransient) {
        let linear_boost = 10.0_f32.powf(self.config.max_boost_db / 20.0);
        let gain = (1.0 + region.severity * (linear_boost - 1.0)).min(linear_boost);
        for i in region.start..region.end.min(output.len()) {
            output[i] = (output[i] * gain).clamp(-1.0, 1.0);
        }
    }

    /// Replace the damaged region with a synthetic transient.
    fn synthetic_replace(&self, output: &mut [f32], region: &DamagedTransient) {
        let len = region.end - region.start;
        if len == 0 {
            return;
        }
        for i in 0..len {
            let idx = region.start + i;
            if idx >= output.len() {
                break;
            }
            // Generate a simple synthetic attack shape
            #[allow(clippy::cast_precision_loss)]
            let t = i as f32 / len as f32;
            let synthetic = region.expected_peak * (1.0 - t) * (-t * 3.0).exp();
            // Blend with original
            output[idx] = (output[idx] * 0.3 + synthetic * 0.7).clamp(-1.0, 1.0);
        }
    }

    /// Get the current configuration.
    pub fn config(&self) -> &TransientRepairConfig {
        &self.config
    }

    /// Update the configuration (recomputes coefficients at given sample rate).
    #[allow(clippy::cast_precision_loss)]
    pub fn set_config(&mut self, config: TransientRepairConfig, sample_rate: u32) {
        let sr = sample_rate as f64;
        self.attack_coeff = ((-1.0_f64) / (config.attack_ms as f64 * 0.001 * sr)).exp() as f32;
        self.release_coeff = ((-1.0_f64) / (config.release_ms as f64 * 0.001 * sr)).exp() as f32;
        self.config = config;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_signal_with_transient(len: usize, transient_pos: usize, peak: f32) -> Vec<f32> {
        let mut signal = vec![0.0_f32; len];
        // Add a transient burst
        let burst_len = 64.min(len - transient_pos);
        for i in 0..burst_len {
            #[allow(clippy::cast_precision_loss)]
            let t = i as f32 / burst_len as f32;
            signal[transient_pos + i] = peak * (1.0 - t) * (-t * 3.0).exp();
        }
        signal
    }

    #[test]
    fn test_default_config() {
        let config = TransientRepairConfig::default();
        assert!((config.detection_threshold - 0.3).abs() < f32::EPSILON);
        assert!((config.attack_ms - 5.0).abs() < f32::EPSILON);
        assert_eq!(config.strategy, RepairStrategy::EnvelopeReshape);
        assert_eq!(config.window_size, 512);
    }

    #[test]
    fn test_create_repairer() {
        let repairer = TransientRepairer::with_defaults(44100);
        assert!(repairer.attack_coeff > 0.0 && repairer.attack_coeff < 1.0);
        assert!(repairer.release_coeff > 0.0 && repairer.release_coeff < 1.0);
    }

    #[test]
    fn test_compute_envelope_silence() {
        let repairer = TransientRepairer::with_defaults(44100);
        let silence = vec![0.0_f32; 1024];
        let env = repairer.compute_envelope(&silence);
        assert_eq!(env.len(), 1024);
        for &v in &env {
            assert!(v.abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_compute_envelope_follows_signal() {
        let repairer = TransientRepairer::with_defaults(44100);
        let signal = make_signal_with_transient(2048, 512, 0.8);
        let env = repairer.compute_envelope(&signal);
        // Envelope should rise near the transient
        let max_before = env[0..500].iter().cloned().fold(0.0_f32, f32::max);
        let max_at = env[512..600].iter().cloned().fold(0.0_f32, f32::max);
        assert!(max_at > max_before);
    }

    #[test]
    fn test_detect_no_damage_on_silence() {
        let repairer = TransientRepairer::with_defaults(44100);
        let silence = vec![0.0_f32; 4096];
        let damaged = repairer.detect(&silence);
        assert!(damaged.is_empty());
    }

    #[test]
    fn test_repair_preserves_length() {
        let repairer = TransientRepairer::with_defaults(44100);
        let signal = vec![0.1_f32; 4096];
        let repaired = repairer.repair(&signal);
        assert_eq!(repaired.len(), signal.len());
    }

    #[test]
    fn test_repair_clamps_output() {
        let config = TransientRepairConfig {
            max_boost_db: 40.0,
            strategy: RepairStrategy::EnergyBoost,
            detection_threshold: 0.01,
            ..TransientRepairConfig::default()
        };
        let repairer = TransientRepairer::new(config, 44100);
        let signal = vec![0.9_f32; 4096];
        let repaired = repairer.repair(&signal);
        for &s in &repaired {
            assert!(s >= -1.0 && s <= 1.0);
        }
    }

    #[test]
    fn test_envelope_reshape_strategy() {
        let config = TransientRepairConfig {
            strategy: RepairStrategy::EnvelopeReshape,
            ..TransientRepairConfig::default()
        };
        let repairer = TransientRepairer::new(config, 44100);
        let signal = vec![0.5_f32; 4096];
        let repaired = repairer.repair(&signal);
        assert_eq!(repaired.len(), 4096);
    }

    #[test]
    fn test_context_interpolation_strategy() {
        let config = TransientRepairConfig {
            strategy: RepairStrategy::ContextInterpolation,
            ..TransientRepairConfig::default()
        };
        let repairer = TransientRepairer::new(config, 44100);
        let signal = vec![0.2_f32; 4096];
        let repaired = repairer.repair(&signal);
        assert_eq!(repaired.len(), 4096);
    }

    #[test]
    fn test_synthetic_replace_strategy() {
        let config = TransientRepairConfig {
            strategy: RepairStrategy::SyntheticReplace,
            ..TransientRepairConfig::default()
        };
        let repairer = TransientRepairer::new(config, 44100);
        let signal = vec![0.3_f32; 4096];
        let repaired = repairer.repair(&signal);
        assert_eq!(repaired.len(), 4096);
    }

    #[test]
    fn test_damaged_transient_fields() {
        let dt = DamagedTransient {
            start: 100,
            end: 200,
            severity: 0.75,
            expected_peak: 0.9,
            actual_peak: 0.2,
        };
        assert_eq!(dt.start, 100);
        assert_eq!(dt.end, 200);
        assert!((dt.severity - 0.75).abs() < f32::EPSILON);
    }

    #[test]
    fn test_set_config() {
        let mut repairer = TransientRepairer::with_defaults(44100);
        let new_config = TransientRepairConfig {
            detection_threshold: 0.5,
            attack_ms: 10.0,
            release_ms: 100.0,
            ..TransientRepairConfig::default()
        };
        repairer.set_config(new_config, 48000);
        assert!((repairer.config().detection_threshold - 0.5).abs() < f32::EPSILON);
        assert!((repairer.config().attack_ms - 10.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_repair_empty_signal() {
        let repairer = TransientRepairer::with_defaults(44100);
        let empty: Vec<f32> = Vec::new();
        let repaired = repairer.repair(&empty);
        assert!(repaired.is_empty());
    }
}
