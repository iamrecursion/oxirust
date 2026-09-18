//! Spectral gating denoiser for `OxiMedia` denoise crate.
//!
//! A spectral gate attenuates frequency-domain bins that fall below a noise
//! threshold, effectively suppressing background noise without affecting
//! signal-dominant bins.

#![allow(dead_code)]

/// A per-bin threshold used by the spectral gate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GateThreshold {
    /// Noise floor estimate for this bin (linear RMS).
    noise_floor: f32,
    /// Multiplier above the noise floor that triggers the gate to open.
    ratio: f32,
}

impl GateThreshold {
    /// Create a new threshold with a noise floor and a ratio multiplier.
    pub fn new(noise_floor: f32, ratio: f32) -> Self {
        Self { noise_floor, ratio }
    }

    /// The absolute threshold value (noise_floor × ratio).
    pub fn threshold_value(&self) -> f32 {
        self.noise_floor * self.ratio
    }

    /// True if `magnitude` exceeds this threshold.
    pub fn is_above_threshold(&self, magnitude: f32) -> bool {
        magnitude > self.threshold_value()
    }
}

/// Configuration for `SpectralGate`.
#[derive(Clone, Debug)]
pub struct SpectralGateConfig {
    /// Attack time in milliseconds (smoothing when gate opens).
    pub attack_ms: f32,
    /// Release time in milliseconds (smoothing when gate closes).
    pub release_ms: f32,
    /// Maximum attenuation in dB when gate is fully closed.
    pub floor_db: f32,
    /// Number of FFT bins (must be > 0).
    pub n_bins: usize,
    /// Ratio above the noise floor at which the gate opens.
    pub threshold_ratio: f32,
}

impl Default for SpectralGateConfig {
    fn default() -> Self {
        Self {
            attack_ms: 5.0,
            release_ms: 50.0,
            floor_db: -80.0,
            n_bins: 512,
            threshold_ratio: 1.5,
        }
    }
}

impl SpectralGateConfig {
    /// Create with a custom bin count.
    pub fn with_bins(n_bins: usize) -> Self {
        Self {
            n_bins,
            ..Default::default()
        }
    }

    /// Attack time in milliseconds.
    pub fn attack_ms(&self) -> f32 {
        self.attack_ms
    }

    /// Release time in milliseconds.
    pub fn release_ms(&self) -> f32 {
        self.release_ms
    }

    /// Floor attenuation when gate is closed, in dB.
    pub fn floor_db(&self) -> f32 {
        self.floor_db
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.n_bins == 0 {
            return Err("n_bins must be greater than 0".to_string());
        }
        if self.attack_ms <= 0.0 {
            return Err("attack_ms must be positive".to_string());
        }
        if self.release_ms <= 0.0 {
            return Err("release_ms must be positive".to_string());
        }
        if self.floor_db > 0.0 {
            return Err("floor_db must be <= 0".to_string());
        }
        if self.threshold_ratio < 1.0 {
            return Err("threshold_ratio must be >= 1.0".to_string());
        }
        Ok(())
    }
}

/// Spectral gate that processes magnitude spectra bin-by-bin.
pub struct SpectralGate {
    config: SpectralGateConfig,
    /// Smoothed gain per bin (0.0–1.0).
    gain: Vec<f32>,
    /// Estimated noise floor per bin.
    noise_floor: Vec<f32>,
    /// Total reduction applied across all frames (for stats).
    cumulative_reduction_db: f64,
    /// Number of frames processed.
    frames_processed: u64,
}

impl SpectralGate {
    /// Construct a new `SpectralGate`.
    pub fn new(config: SpectralGateConfig) -> Self {
        let n = config.n_bins;
        Self {
            config,
            gain: vec![1.0; n],
            noise_floor: vec![1e-6; n],
            cumulative_reduction_db: 0.0,
            frames_processed: 0,
        }
    }

    /// Set the estimated noise floor for all bins from a noise-only magnitude spectrum.
    pub fn learn_noise_floor(&mut self, magnitudes: &[f32]) {
        let n = self.config.n_bins.min(magnitudes.len());
        for (floor, &mag) in self.noise_floor.iter_mut().zip(magnitudes[..n].iter()) {
            *floor = mag.max(1e-9);
        }
    }

    /// Process one frame of complex-pair magnitude data.
    ///
    /// `magnitudes` contains linear magnitudes for each bin.
    /// `output` receives gain-multiplied magnitudes (same length as `magnitudes`).
    #[allow(clippy::cast_precision_loss)]
    pub fn process_frame(&mut self, magnitudes: &[f32], output: &mut [f32]) {
        let n = self.config.n_bins.min(magnitudes.len()).min(output.len());
        let floor_lin = 10.0_f32.powf(self.config.floor_db / 20.0);
        let ratio = self.config.threshold_ratio;

        let mut frame_reduction = 0.0_f64;

        for i in 0..n {
            let threshold = self.noise_floor[i] * ratio;
            let target_gain = if magnitudes[i] > threshold {
                1.0
            } else {
                floor_lin
            };
            // Simple smoothing
            self.gain[i] = self.gain[i] * 0.9 + target_gain * 0.1;
            output[i] = magnitudes[i] * self.gain[i];
            if self.gain[i] < 1.0 {
                frame_reduction += (1.0 - self.gain[i] as f64).abs() * 20.0;
            }
        }
        self.cumulative_reduction_db += frame_reduction / n as f64;
        self.frames_processed += 1;
    }

    /// Average reduction in dB across all frames processed.
    #[allow(clippy::cast_precision_loss)]
    pub fn reduction_db(&self) -> f64 {
        if self.frames_processed == 0 {
            0.0
        } else {
            self.cumulative_reduction_db / self.frames_processed as f64
        }
    }

    /// Reset gate state.
    pub fn reset(&mut self) {
        self.gain.iter_mut().for_each(|g| *g = 1.0);
        self.noise_floor.iter_mut().for_each(|f| *f = 1e-6);
        self.cumulative_reduction_db = 0.0;
        self.frames_processed = 0;
    }

    /// Configuration accessor.
    pub fn config(&self) -> &SpectralGateConfig {
        &self.config
    }
}

/// Statistics collected from a `SpectralGate` run.
#[derive(Clone, Debug)]
pub struct SpectralGateStats {
    /// Average reduction in dB.
    pub avg_reduction_db: f64,
    /// Total frames processed.
    pub frames_processed: u64,
}

impl SpectralGateStats {
    /// Collect stats from a running gate.
    pub fn from_gate(gate: &SpectralGate) -> Self {
        Self {
            avg_reduction_db: gate.reduction_db(),
            frames_processed: gate.frames_processed,
        }
    }

    /// Average spectral reduction in dB.
    pub fn avg_reduction(&self) -> f64 {
        self.avg_reduction_db
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gate_threshold_is_above() {
        let t = GateThreshold::new(0.01, 2.0);
        assert!(t.is_above_threshold(0.025));
    }

    #[test]
    fn test_gate_threshold_is_below() {
        let t = GateThreshold::new(0.01, 2.0);
        assert!(!t.is_above_threshold(0.015));
    }

    #[test]
    fn test_gate_threshold_value() {
        let t = GateThreshold::new(0.1, 3.0);
        assert!((t.threshold_value() - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_config_default_attack_ms() {
        let cfg = SpectralGateConfig::default();
        assert!((cfg.attack_ms() - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_config_default_release_ms() {
        let cfg = SpectralGateConfig::default();
        assert!((cfg.release_ms() - 50.0).abs() < 1e-5);
    }

    #[test]
    fn test_config_floor_db() {
        let cfg = SpectralGateConfig::default();
        assert!(cfg.floor_db() < 0.0);
    }

    #[test]
    fn test_config_validate_ok() {
        let cfg = SpectralGateConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_validate_zero_bins() {
        let mut cfg = SpectralGateConfig::default();
        cfg.n_bins = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validate_bad_floor_db() {
        let mut cfg = SpectralGateConfig::default();
        cfg.floor_db = 3.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_gate_process_frame_output_length() {
        let cfg = SpectralGateConfig::with_bins(8);
        let mut gate = SpectralGate::new(cfg);
        let mags = vec![0.5_f32; 8];
        let mut out = vec![0.0_f32; 8];
        gate.process_frame(&mags, &mut out);
        assert_eq!(out.len(), 8);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_gate_learn_noise_floor_and_open() {
        let cfg = SpectralGateConfig::with_bins(4);
        let mut gate = SpectralGate::new(cfg);
        gate.learn_noise_floor(&[0.01, 0.01, 0.01, 0.01]);
        let mags = vec![1.0_f32; 4]; // far above threshold
        let mut out = vec![0.0_f32; 4];
        gate.process_frame(&mags, &mut out);
        // With gate open, output should be close to input
        assert!(out.iter().all(|&v| v > 0.5));
    }

    #[test]
    fn test_gate_reduction_db_zero_before_processing() {
        let cfg = SpectralGateConfig::default();
        let gate = SpectralGate::new(cfg);
        assert!((gate.reduction_db() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_gate_reset() {
        let cfg = SpectralGateConfig::with_bins(4);
        let mut gate = SpectralGate::new(cfg);
        let mags = vec![1e-8_f32; 4];
        let mut out = vec![0.0_f32; 4];
        gate.process_frame(&mags, &mut out);
        gate.reset();
        assert_eq!(gate.frames_processed, 0);
    }

    #[test]
    fn test_stats_avg_reduction() {
        let cfg = SpectralGateConfig::with_bins(4);
        let mut gate = SpectralGate::new(cfg);
        let mags = vec![1e-9_f32; 4];
        let mut out = vec![0.0_f32; 4];
        gate.process_frame(&mags, &mut out);
        let stats = SpectralGateStats::from_gate(&gate);
        assert!(stats.avg_reduction() >= 0.0);
    }

    // ----- Known-noise spectral gate tests -----

    #[test]
    fn test_spectral_gate_learns_noise_floor() {
        // Uniform low-level noise learned as the noise floor.
        let cfg = SpectralGateConfig {
            n_bins: 512,
            threshold_ratio: 1.5,
            ..Default::default()
        };
        let mut gate = SpectralGate::new(cfg);
        let noise_mags = vec![0.1f32; 512];
        gate.learn_noise_floor(&noise_mags);

        // Process the same noise-only input — all bins are at or below the noise floor,
        // so the gate should attenuate and report a positive reduction.
        let mut out = vec![0.0f32; 512];
        gate.process_frame(&noise_mags, &mut out);

        assert!(
            gate.reduction_db() > 0.0,
            "gate should report positive reduction after processing noise-level input, got {}",
            gate.reduction_db()
        );
    }

    #[test]
    fn test_spectral_gate_signal_bins_survive() {
        // Learn a low noise floor, then inject strong signal at three bins.
        let cfg = SpectralGateConfig {
            n_bins: 512,
            threshold_ratio: 1.5,
            ..Default::default()
        };
        let mut gate = SpectralGate::new(cfg);
        let noise_floor = vec![0.1f32; 512];
        gate.learn_noise_floor(&noise_floor);

        // Build mixed spectrum: mostly noise-level, with 3 strong signal bins.
        let mut mixed = vec![0.1f32; 512];
        mixed[100] = 10.0;
        mixed[200] = 10.0;
        mixed[300] = 10.0;

        let mut out = vec![0.0f32; 512];
        gate.process_frame(&mixed, &mut out);

        // Signal bins must survive (output ≥ 50% of input).
        assert!(
            out[100] > mixed[100] * 0.5,
            "bin 100 (signal) should largely pass through: got {} from {}",
            out[100],
            mixed[100]
        );
        assert!(
            out[200] > mixed[200] * 0.5,
            "bin 200 (signal) should largely pass through: got {} from {}",
            out[200],
            mixed[200]
        );
        assert!(
            out[300] > mixed[300] * 0.5,
            "bin 300 (signal) should largely pass through: got {} from {}",
            out[300],
            mixed[300]
        );
    }

    #[test]
    fn test_spectral_gate_reduction_db_positive() {
        // After learning the noise floor and processing the same noise-only spectrum,
        // reduction_db() must be positive (gate is suppressing).
        let cfg = SpectralGateConfig {
            n_bins: 512,
            threshold_ratio: 1.5,
            ..Default::default()
        };
        let mut gate = SpectralGate::new(cfg);
        let noise_mags = vec![0.1f32; 512];
        gate.learn_noise_floor(&noise_mags);

        let mut out = vec![0.0f32; 512];
        // Process multiple frames so smoothed gain settles into attenuation.
        for _ in 0..5 {
            gate.process_frame(&noise_mags, &mut out);
        }

        assert!(
            gate.reduction_db() > 0.0,
            "reduction_db() should be > 0.0 after gating noise-only signal; got {}",
            gate.reduction_db()
        );
    }
}
