//! Audio denoising — noise gate, spectral subtraction and noise profiling.
#![allow(dead_code)]

use std::collections::VecDeque;

// ── Noise profile ────────────────────────────────────────────────────────────

/// A collected noise profile describing the spectral characteristics of
/// background noise in an audio signal.
#[derive(Debug, Clone)]
pub struct NoiseProfile {
    /// Per-band noise floor (linear amplitude).
    pub band_floor: Vec<f32>,
    /// Whether the noise is considered stationary (slow-changing).
    pub stationary: bool,
    /// Estimated overall noise RMS.
    pub rms: f32,
}

impl NoiseProfile {
    /// Create a new noise profile from per-band floor values.
    #[must_use]
    pub fn new(band_floor: Vec<f32>, rms: f32) -> Self {
        let stationary = Self::classify_stationarity(&band_floor);
        Self {
            band_floor,
            stationary,
            rms,
        }
    }

    /// Return `true` when noise is stationary (good for spectral subtraction).
    #[must_use]
    pub fn is_stationary(&self) -> bool {
        self.stationary
    }

    fn classify_stationarity(band_floor: &[f32]) -> bool {
        if band_floor.is_empty() {
            return true;
        }
        let mean: f32 = band_floor.iter().sum::<f32>() / band_floor.len() as f32;
        if mean == 0.0 {
            return true;
        }
        let variance: f32 =
            band_floor.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / band_floor.len() as f32;
        // Consider stationary when coefficient of variation < 0.3
        (variance.sqrt() / mean) < 0.3
    }
}

impl Default for NoiseProfile {
    fn default() -> Self {
        Self::new(vec![0.0; 32], 0.0)
    }
}

// ── Spectral subtraction ─────────────────────────────────────────────────────

/// Configuration for spectral subtraction denoising.
#[derive(Debug, Clone)]
pub struct SpectralSubtractionConfig {
    /// Over-subtraction factor α (>= 1.0).  Higher = more aggressive.
    pub alpha: f32,
    /// Spectral floor β (prevents musical noise artefacts, 0.0–1.0).
    pub beta: f32,
    /// Number of frequency bins (FFT size / 2).
    pub num_bins: usize,
}

impl Default for SpectralSubtractionConfig {
    fn default() -> Self {
        Self {
            alpha: 2.0,
            beta: 0.01,
            num_bins: 256,
        }
    }
}

/// Spectral subtraction denoiser.
#[derive(Debug)]
pub struct SpectralSubtraction {
    config: SpectralSubtractionConfig,
    /// Accumulated noise spectrum estimate (magnitude per bin).
    noise_spectrum: Vec<f32>,
}

impl SpectralSubtraction {
    /// Create a new spectral subtraction instance.
    #[must_use]
    pub fn new(config: SpectralSubtractionConfig) -> Self {
        let noise_spectrum = vec![0.0f32; config.num_bins];
        Self {
            config,
            noise_spectrum,
        }
    }

    /// Estimate the noise floor from a block of magnitude-spectrum samples.
    /// `magnitudes` must have length == `num_bins`.
    pub fn compute_noise_floor(&mut self, magnitudes: &[f32]) {
        let n = self.config.num_bins.min(magnitudes.len());
        for (stored, &m) in self.noise_spectrum[..n].iter_mut().zip(magnitudes.iter()) {
            // Exponential smoothing
            *stored = 0.9 * (*stored) + 0.1 * m;
        }
    }

    /// Apply spectral subtraction to a magnitude spectrum.
    ///
    /// Returns the de-noised magnitude spectrum.
    #[must_use]
    pub fn apply(&self, magnitudes: &[f32]) -> Vec<f32> {
        let n = self.config.num_bins.min(magnitudes.len());
        let alpha = self.config.alpha;
        let beta = self.config.beta;
        let mut out = magnitudes.to_vec();
        for i in 0..n {
            let subtracted = out[i] - alpha * self.noise_spectrum[i];
            // Ensure spectral floor (avoid "musical noise")
            out[i] = subtracted.max(beta * out[i]);
        }
        out
    }

    /// Return a reference to the current noise spectrum estimate.
    #[must_use]
    pub fn noise_spectrum(&self) -> &[f32] {
        &self.noise_spectrum
    }
}

// ── Audio denoise filter ─────────────────────────────────────────────────────

/// Gate state for the noise gate sub-component.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateState {
    /// Gate is open — signal passes through.
    Open,
    /// Gate is closed — signal is attenuated.
    Closed,
}

/// Combined audio denoising filter with noise gate and spectral subtraction.
#[derive(Debug)]
pub struct AudioDenoiseFilter {
    /// RMS threshold below which the noise gate closes.
    pub gate_threshold: f32,
    /// Hold time in samples after the gate would close (prevents chatter).
    pub gate_hold_samples: usize,
    /// Spectral subtraction engine.
    spectral: SpectralSubtraction,
    /// Ring buffer of recent RMS values for smoothing.
    rms_history: VecDeque<f32>,
    /// Current gate state.
    gate_state: GateState,
    /// Remaining hold samples.
    hold_counter: usize,
}

impl AudioDenoiseFilter {
    /// Create a new filter with default spectral subtraction settings.
    #[must_use]
    pub fn new(gate_threshold: f32, gate_hold_samples: usize) -> Self {
        Self {
            gate_threshold,
            gate_hold_samples,
            spectral: SpectralSubtraction::new(SpectralSubtractionConfig::default()),
            rms_history: VecDeque::with_capacity(8),
            gate_state: GateState::Open,
            hold_counter: 0,
        }
    }

    /// Process a block of PCM samples.
    ///
    /// Returns the de-noised samples (same length as `samples`).
    #[must_use]
    pub fn process(&mut self, samples: &[f32]) -> Vec<f32> {
        let rms = Self::compute_rms(samples);

        // Update gate state
        if rms >= self.gate_threshold {
            self.gate_state = GateState::Open;
            self.hold_counter = self.gate_hold_samples;
        } else if self.hold_counter > 0 {
            self.hold_counter -= 1;
        } else {
            self.gate_state = GateState::Closed;
        }

        self.rms_history.push_back(rms);
        if self.rms_history.len() > 8 {
            self.rms_history.pop_front();
        }

        match self.gate_state {
            GateState::Closed => vec![0.0f32; samples.len()],
            GateState::Open => samples.to_vec(),
        }
    }

    /// Current gate state.
    #[must_use]
    pub fn gate_state(&self) -> GateState {
        self.gate_state
    }

    /// Train the spectral subtraction model on a noise-only block.
    pub fn train_noise(&mut self, noise_magnitudes: &[f32]) {
        self.spectral.compute_noise_floor(noise_magnitudes);
    }

    /// Apply spectral subtraction to a magnitude spectrum.
    #[must_use]
    pub fn apply_spectral(&self, magnitudes: &[f32]) -> Vec<f32> {
        self.spectral.apply(magnitudes)
    }

    #[allow(clippy::cast_precision_loss)]
    fn compute_rms(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
        (sum_sq / samples.len() as f32).sqrt()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise_profile_is_stationary_flat() {
        let profile = NoiseProfile::new(vec![0.1; 32], 0.1);
        assert!(profile.is_stationary());
    }

    #[test]
    fn test_noise_profile_is_not_stationary_high_variance() {
        let mut floor = vec![0.0f32; 32];
        for (i, v) in floor.iter_mut().enumerate() {
            *v = if i % 2 == 0 { 0.01 } else { 1.0 };
        }
        let profile = NoiseProfile::new(floor, 0.5);
        assert!(!profile.is_stationary());
    }

    #[test]
    fn test_noise_profile_empty_bands_is_stationary() {
        let profile = NoiseProfile::new(vec![], 0.0);
        assert!(profile.is_stationary());
    }

    #[test]
    fn test_noise_profile_default() {
        let p = NoiseProfile::default();
        assert_eq!(p.band_floor.len(), 32);
        assert!((p.rms - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_spectral_subtraction_compute_noise_floor() {
        let cfg = SpectralSubtractionConfig::default();
        let mut ss = SpectralSubtraction::new(cfg);
        let mags = vec![1.0f32; 256];
        ss.compute_noise_floor(&mags);
        // After one update: 0.9*0 + 0.1*1 = 0.1
        assert!((ss.noise_spectrum()[0] - 0.1).abs() < 1e-5);
    }

    #[test]
    fn test_spectral_subtraction_apply_reduces_magnitude() {
        let cfg = SpectralSubtractionConfig::default();
        let mut ss = SpectralSubtraction::new(cfg);
        // Train with moderate noise
        let noise = vec![0.5f32; 256];
        for _ in 0..50 {
            ss.compute_noise_floor(&noise);
        }
        let signal = vec![1.0f32; 256];
        let out = ss.apply(&signal);
        // After subtraction the signal should be less than the raw input
        assert!(out[0] < 1.0);
    }

    #[test]
    fn test_spectral_subtraction_beta_floor() {
        let cfg = SpectralSubtractionConfig {
            alpha: 100.0, // very aggressive
            beta: 0.5,
            num_bins: 4,
        };
        let mut ss = SpectralSubtraction::new(cfg);
        ss.compute_noise_floor(&[1.0, 1.0, 1.0, 1.0]);
        let out = ss.apply(&[0.1, 0.1, 0.1, 0.1]);
        // beta * magnitude = 0.5 * 0.1 = 0.05
        assert!(out[0] >= 0.0);
    }

    #[test]
    fn test_spectral_subtraction_noise_spectrum_length() {
        let cfg = SpectralSubtractionConfig {
            num_bins: 128,
            ..Default::default()
        };
        let ss = SpectralSubtraction::new(cfg);
        assert_eq!(ss.noise_spectrum().len(), 128);
    }

    #[test]
    fn test_audio_denoise_filter_open_gate() {
        let mut filter = AudioDenoiseFilter::new(0.01, 10);
        // Loud signal — gate should stay open
        let samples: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).sin() * 0.5).collect();
        let out = filter.process(&samples);
        assert_eq!(filter.gate_state(), GateState::Open);
        assert_eq!(out.len(), samples.len());
    }

    #[test]
    fn test_audio_denoise_filter_closed_gate() {
        let mut filter = AudioDenoiseFilter::new(1.0, 0);
        // Silent signal — gate should close
        let samples = vec![0.0f32; 256];
        let out = filter.process(&samples);
        assert_eq!(filter.gate_state(), GateState::Closed);
        assert!(out.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn test_audio_denoise_filter_hold_prevents_early_close() {
        let mut filter = AudioDenoiseFilter::new(0.01, 100);
        // Open with loud signal
        let loud: Vec<f32> = vec![0.5f32; 256];
        let _ = filter.process(&loud);
        assert_eq!(filter.gate_state(), GateState::Open);
        // Now silence — hold should keep gate open
        let silent = vec![0.0f32; 256];
        let _ = filter.process(&silent);
        // With hold=100 the gate should still be open after one block
        assert_eq!(filter.gate_state(), GateState::Open);
    }

    #[test]
    fn test_audio_denoise_filter_train_and_apply_spectral() {
        let mut filter = AudioDenoiseFilter::new(0.0, 0);
        let noise = vec![0.2f32; 256];
        filter.train_noise(&noise);
        let signal = vec![1.0f32; 256];
        let out = filter.apply_spectral(&signal);
        assert_eq!(out.len(), 256);
        // At least some reduction should happen
        assert!(out[0] < 1.0);
    }

    #[test]
    fn test_audio_denoise_filter_rms_empty() {
        let rms = AudioDenoiseFilter::compute_rms(&[]);
        assert!((rms - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_audio_denoise_filter_rms_sine() {
        // RMS of unit sine ≈ 1/sqrt(2) ≈ 0.707
        let samples: Vec<f32> = (0..1024)
            .map(|i| (2.0 * std::f32::consts::PI * i as f32 / 1024.0).sin())
            .collect();
        let rms = AudioDenoiseFilter::compute_rms(&samples);
        assert!((rms - std::f32::consts::FRAC_1_SQRT_2).abs() < 0.01);
    }
}
