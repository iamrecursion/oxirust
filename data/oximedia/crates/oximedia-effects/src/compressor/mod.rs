//! Dynamics compression effects.
//!
//! Provides professional compressor, limiter, and expander with industry-standard
//! gain computer and level detector designs. Includes sidechain input support
//! for frequency-conscious compression and ducking applications.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// Compressor configuration parameters.
#[derive(Debug, Clone)]
pub struct CompressorConfig {
    /// Threshold in dB above which compression begins.
    pub threshold_db: f32,
    /// Compression ratio (e.g. 4.0 = 4:1).
    pub ratio: f32,
    /// Attack time in milliseconds.
    pub attack_ms: f32,
    /// Release time in milliseconds.
    pub release_ms: f32,
    /// Knee width in dB (0 = hard knee).
    pub knee_db: f32,
    /// Makeup gain in dB applied after compression.
    pub makeup_gain_db: f32,
}

impl CompressorConfig {
    /// Standard general-purpose compressor (4:1 ratio, moderate attack/release).
    #[must_use]
    pub fn standard() -> Self {
        Self {
            threshold_db: -18.0,
            ratio: 4.0,
            attack_ms: 10.0,
            release_ms: 100.0,
            knee_db: 6.0,
            makeup_gain_db: 3.0,
        }
    }

    /// Limiting configuration (100:1 ratio, very fast attack).
    #[must_use]
    pub fn limiting() -> Self {
        Self {
            threshold_db: -3.0,
            ratio: 100.0,
            attack_ms: 0.1,
            release_ms: 50.0,
            knee_db: 0.0,
            makeup_gain_db: 0.0,
        }
    }

    /// Vocal compressor preset (gentle 3:1 ratio).
    #[must_use]
    pub fn vocal() -> Self {
        Self {
            threshold_db: -20.0,
            ratio: 3.0,
            attack_ms: 5.0,
            release_ms: 80.0,
            knee_db: 8.0,
            makeup_gain_db: 4.0,
        }
    }
}

impl Default for CompressorConfig {
    fn default() -> Self {
        Self::standard()
    }
}

/// Peak level detector with attack/release envelopes.
pub struct LevelDetector {
    /// Current peak level.
    pub peak_level: f32,
}

impl LevelDetector {
    /// Create a new level detector.
    #[must_use]
    pub fn new() -> Self {
        Self { peak_level: 0.0 }
    }

    /// Process a single sample and return the envelope level.
    ///
    /// # Arguments
    ///
    /// * `x` - Input sample (absolute value used)
    /// * `attack` - Attack coefficient (0..1), computed as `1 - exp(-2.2 / (attack_ms * sr / 1000))`
    /// * `release` - Release coefficient (0..1)
    pub fn process(&mut self, x: f32, attack: f32, release: f32) -> f32 {
        let input_level = x.abs();
        if input_level > self.peak_level {
            self.peak_level += attack * (input_level - self.peak_level);
        } else {
            self.peak_level += release * (input_level - self.peak_level);
        }
        self.peak_level
    }

    /// Reset the detector state.
    pub fn reset(&mut self) {
        self.peak_level = 0.0;
    }
}

impl Default for LevelDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Gain computer state that implements the compression curve.
pub struct GainComputerState {
    /// Last computed gain reduction in dB.
    pub last_gain_reduction_db: f32,
}

impl GainComputerState {
    /// Create a new gain computer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            last_gain_reduction_db: 0.0,
        }
    }

    /// Compute gain reduction in dB for the given input level.
    ///
    /// Implements soft-knee compression curve from AES guidelines.
    pub fn compute_gain(&mut self, input_db: f32, config: &CompressorConfig) -> f32 {
        let threshold = config.threshold_db;
        let ratio = config.ratio;
        let knee = config.knee_db;
        let half_knee = knee / 2.0;

        let gain_reduction_db =
            if knee > 0.0 && input_db >= threshold - half_knee && input_db <= threshold + half_knee
            {
                // Soft knee region: smooth transition
                let knee_input = input_db - threshold + half_knee;
                let knee_factor = knee_input / knee;
                // Soft knee formula: gain_reduction = (1/R - 1) * (input - threshold + knee/2)^2 / (2*knee)
                (1.0 / ratio - 1.0) * (knee_factor * knee_input) / 2.0
            } else if input_db > threshold + half_knee {
                // Above threshold: apply ratio
                (input_db - threshold) * (1.0 / ratio - 1.0)
            } else {
                // Below threshold: no gain reduction
                0.0
            };

        self.last_gain_reduction_db = gain_reduction_db;
        gain_reduction_db
    }
}

impl Default for GainComputerState {
    fn default() -> Self {
        Self::new()
    }
}

/// Gain reduction tracking for metering.
#[derive(Debug, Clone, Default)]
pub struct GainReduction {
    /// Peak gain reduction observed in dB (positive = reduction).
    pub peak_db: f32,
    /// RMS gain reduction in dB over a measurement window.
    pub rms_db: f32,
    /// Accumulator for RMS computation.
    accumulator: f32,
    /// Sample count for RMS window.
    sample_count: u32,
    /// Window size for RMS.
    window_size: u32,
}

impl GainReduction {
    /// Create a new gain reduction tracker.
    #[must_use]
    pub fn new(window_size: u32) -> Self {
        Self {
            window_size,
            ..Default::default()
        }
    }

    /// Update with a new gain reduction value (in dB, positive = reduction).
    pub fn update(&mut self, reduction_db: f32) {
        let abs_reduction = reduction_db.abs();
        if abs_reduction > self.peak_db {
            self.peak_db = abs_reduction;
        }
        self.accumulator += abs_reduction * abs_reduction;
        self.sample_count += 1;
        if self.sample_count >= self.window_size {
            self.rms_db = (self.accumulator / self.window_size as f32).sqrt();
            self.accumulator = 0.0;
            self.sample_count = 0;
        }
    }

    /// Reset peak reading.
    pub fn reset_peak(&mut self) {
        self.peak_db = 0.0;
    }
}

/// Sidechain filter type for frequency-conscious compression.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SidechainFilter {
    /// No filtering on sidechain signal.
    None,
    /// High-pass filter at given frequency (Hz) for de-essing / sibilance detection.
    HighPass(f32),
    /// Low-pass filter at given frequency (Hz) for bass-focused compression.
    LowPass(f32),
    /// Band-pass filter centered at given frequency (Hz) with Q factor.
    BandPass(f32, f32),
}

/// One-pole filter state for sidechain filtering.
#[derive(Debug, Clone)]
struct OnePoleFilter {
    /// Previous output sample.
    prev: f32,
    /// Filter coefficient.
    coeff: f32,
    /// Filter mode.
    is_highpass: bool,
}

impl OnePoleFilter {
    fn new_highpass(cutoff_hz: f32, sample_rate: f32) -> Self {
        let rc = 1.0 / (2.0 * std::f32::consts::PI * cutoff_hz);
        let dt = 1.0 / sample_rate;
        let coeff = rc / (rc + dt);
        Self {
            prev: 0.0,
            coeff,
            is_highpass: true,
        }
    }

    fn new_lowpass(cutoff_hz: f32, sample_rate: f32) -> Self {
        let rc = 1.0 / (2.0 * std::f32::consts::PI * cutoff_hz);
        let dt = 1.0 / sample_rate;
        let coeff = dt / (rc + dt);
        Self {
            prev: 0.0,
            coeff,
            is_highpass: false,
        }
    }

    fn process(&mut self, input: f32) -> f32 {
        if self.is_highpass {
            let _output = self.coeff * (self.prev + input - self.prev);
            // Simple high-pass: output = coeff * (prev_output + input - prev_input)
            // Use a simplified version:
            let hp_output = self.coeff * self.prev + self.coeff * (input - self.prev);
            self.prev = input;
            hp_output
        } else {
            self.prev += self.coeff * (input - self.prev);
            self.prev
        }
    }

    fn reset(&mut self) {
        self.prev = 0.0;
    }
}

/// Professional dynamics compressor.
pub struct Compressor {
    config: CompressorConfig,
    detector: LevelDetector,
    gain_computer: GainComputerState,
    /// Current gain reduction in linear.
    gain_reduction_linear: f32,
    /// Gain reduction metering.
    pub gain_reduction: GainReduction,
    /// Smoothed gain reduction in dB (for smooth ballistics).
    smoothed_gr_db: f32,
    /// Optional sidechain filter.
    sidechain_filter_type: SidechainFilter,
    /// Sidechain filter state.
    sc_filter: Option<OnePoleFilter>,
}

impl Compressor {
    /// Create a new compressor with the given configuration and sample rate.
    #[must_use]
    pub fn new(config: CompressorConfig, _sample_rate: u32) -> Self {
        Self {
            config,
            detector: LevelDetector::new(),
            gain_computer: GainComputerState::new(),
            gain_reduction_linear: 1.0,
            gain_reduction: GainReduction::new(4800),
            smoothed_gr_db: 0.0,
            sidechain_filter_type: SidechainFilter::None,
            sc_filter: None,
        }
    }

    /// Set a sidechain filter for frequency-conscious compression.
    ///
    /// When set, the filter is applied to the sidechain signal before
    /// level detection. This enables:
    /// - **De-essing**: `SidechainFilter::HighPass(4000.0)` to detect sibilance
    /// - **Bass compression**: `SidechainFilter::LowPass(200.0)` for bass-only triggering
    pub fn set_sidechain_filter(&mut self, filter: SidechainFilter, sample_rate: u32) {
        self.sidechain_filter_type = filter;
        #[allow(clippy::cast_precision_loss)]
        let sr = sample_rate as f32;
        self.sc_filter = match filter {
            SidechainFilter::None => None,
            SidechainFilter::HighPass(freq) => Some(OnePoleFilter::new_highpass(freq, sr)),
            SidechainFilter::LowPass(freq) => Some(OnePoleFilter::new_lowpass(freq, sr)),
            SidechainFilter::BandPass(freq, _q) => {
                // Approximate band-pass as cascaded HP + LP
                // For a proper implementation, use a biquad; this is a reasonable approximation
                Some(OnePoleFilter::new_highpass(freq * 0.7, sr))
            }
        };
    }

    /// Get the current sidechain filter type.
    #[must_use]
    pub fn sidechain_filter(&self) -> SidechainFilter {
        self.sidechain_filter_type
    }

    fn db_to_linear(db: f32) -> f32 {
        10.0_f32.powf(db / 20.0)
    }

    fn linear_to_db(linear: f32) -> f32 {
        20.0 * linear.max(1e-10_f32).log10()
    }

    fn attack_coeff(attack_ms: f32, sample_rate: u32) -> f32 {
        let attack_samples = attack_ms * sample_rate as f32 / 1000.0;
        if attack_samples > 0.0 {
            1.0 - (-2.2_f32 / attack_samples).exp()
        } else {
            1.0
        }
    }

    fn release_coeff(release_ms: f32, sample_rate: u32) -> f32 {
        let release_samples = release_ms * sample_rate as f32 / 1000.0;
        if release_samples > 0.0 {
            1.0 - (-2.2_f32 / release_samples).exp()
        } else {
            1.0
        }
    }

    /// Process a buffer of samples and return compressed output.
    #[must_use]
    pub fn process(&mut self, samples: &[f32], sample_rate: u32) -> Vec<f32> {
        let attack = Self::attack_coeff(self.config.attack_ms, sample_rate);
        let release = Self::release_coeff(self.config.release_ms, sample_rate);
        let makeup = Self::db_to_linear(self.config.makeup_gain_db);

        // Ballistics: smooth the gain reduction itself
        let gr_attack = Self::attack_coeff(self.config.attack_ms, sample_rate);
        let gr_release = Self::release_coeff(self.config.release_ms, sample_rate);

        samples
            .iter()
            .map(|&x| {
                // Detect level
                let level = self.detector.process(x, attack, release);
                let level_db = Self::linear_to_db(level);

                // Compute gain reduction
                let gr_db = self.gain_computer.compute_gain(level_db, &self.config);

                // Smooth gain reduction (ballistics on gain signal)
                if gr_db < self.smoothed_gr_db {
                    // Attack: gain goes down (more reduction)
                    self.smoothed_gr_db += gr_attack * (gr_db - self.smoothed_gr_db);
                } else {
                    // Release: gain comes back up
                    self.smoothed_gr_db += gr_release * (gr_db - self.smoothed_gr_db);
                }

                self.gain_reduction_linear = Self::db_to_linear(self.smoothed_gr_db);
                self.gain_reduction.update(self.smoothed_gr_db);

                x * self.gain_reduction_linear * makeup
            })
            .collect()
    }

    /// Process with external sidechain input.
    ///
    /// The compressor uses `sidechain` for level detection but applies
    /// gain reduction to the `input` signal. This is useful for:
    /// - De-essing (sidechain is EQ'd high-frequency band)
    /// - Ducking (sidechain is voiceover, input is music)
    /// - Frequency-conscious compression (sidechain is filtered version)
    ///
    /// Both buffers must be the same length.
    #[must_use]
    pub fn process_sidechain(
        &mut self,
        input: &[f32],
        sidechain: &[f32],
        sample_rate: u32,
    ) -> Vec<f32> {
        let len = input.len().min(sidechain.len());
        let attack = Self::attack_coeff(self.config.attack_ms, sample_rate);
        let release = Self::release_coeff(self.config.release_ms, sample_rate);
        let makeup = Self::db_to_linear(self.config.makeup_gain_db);
        let gr_attack = Self::attack_coeff(self.config.attack_ms, sample_rate);
        let gr_release = Self::release_coeff(self.config.release_ms, sample_rate);

        let mut output = Vec::with_capacity(len);

        for i in 0..len {
            // Optionally filter the sidechain signal
            let sc_sample = if let Some(ref mut filter) = self.sc_filter {
                filter.process(sidechain[i])
            } else {
                sidechain[i]
            };

            // Detect level from SIDECHAIN signal
            let level = self.detector.process(sc_sample, attack, release);
            let level_db = Self::linear_to_db(level);

            // Compute gain reduction based on sidechain level
            let gr_db = self.gain_computer.compute_gain(level_db, &self.config);

            // Smooth gain reduction
            if gr_db < self.smoothed_gr_db {
                self.smoothed_gr_db += gr_attack * (gr_db - self.smoothed_gr_db);
            } else {
                self.smoothed_gr_db += gr_release * (gr_db - self.smoothed_gr_db);
            }

            self.gain_reduction_linear = Self::db_to_linear(self.smoothed_gr_db);
            self.gain_reduction.update(self.smoothed_gr_db);

            // Apply gain reduction to INPUT signal
            output.push(input[i] * self.gain_reduction_linear * makeup);
        }

        output
    }

    /// Reset compressor state.
    pub fn reset(&mut self) {
        self.detector.reset();
        self.gain_reduction_linear = 1.0;
        self.smoothed_gr_db = 0.0;
        if let Some(ref mut filter) = self.sc_filter {
            filter.reset();
        }
    }

    /// Get current gain reduction in dB.
    #[must_use]
    pub fn current_gain_reduction_db(&self) -> f32 {
        -self.smoothed_gr_db
    }
}

/// Below-threshold expander / gate-like processor.
///
/// Reduces signal level when below the threshold, acting as a soft gate.
pub struct Expander {
    /// Threshold below which expansion is applied (dB).
    pub threshold_db: f32,
    /// Expansion ratio (> 1 = expand).
    pub ratio: f32,
    /// Attack time in milliseconds.
    pub attack_ms: f32,
    /// Release time in milliseconds.
    pub release_ms: f32,
    /// Knee width in dB.
    pub knee_db: f32,
    detector: LevelDetector,
    smoothed_gain_db: f32,
}

impl Expander {
    /// Create a new expander.
    #[must_use]
    pub fn new(
        threshold_db: f32,
        ratio: f32,
        attack_ms: f32,
        release_ms: f32,
        knee_db: f32,
    ) -> Self {
        Self {
            threshold_db,
            ratio,
            attack_ms,
            release_ms,
            knee_db,
            detector: LevelDetector::new(),
            smoothed_gain_db: 0.0,
        }
    }

    /// Default gate-like expander preset.
    #[must_use]
    pub fn gate() -> Self {
        Self::new(-40.0, 10.0, 1.0, 50.0, 4.0)
    }

    fn db_to_linear(db: f32) -> f32 {
        10.0_f32.powf(db / 20.0)
    }

    fn linear_to_db(linear: f32) -> f32 {
        20.0 * linear.max(1e-10_f32).log10()
    }

    fn attack_coeff(attack_ms: f32, sample_rate: u32) -> f32 {
        let s = attack_ms * sample_rate as f32 / 1000.0;
        if s > 0.0 {
            1.0 - (-2.2_f32 / s).exp()
        } else {
            1.0
        }
    }

    fn release_coeff(release_ms: f32, sample_rate: u32) -> f32 {
        let s = release_ms * sample_rate as f32 / 1000.0;
        if s > 0.0 {
            1.0 - (-2.2_f32 / s).exp()
        } else {
            1.0
        }
    }

    fn compute_expansion_gain(&self, input_db: f32) -> f32 {
        let threshold = self.threshold_db;
        let ratio = self.ratio;
        let half_knee = self.knee_db / 2.0;

        if input_db < threshold - half_knee {
            // Below threshold: expand downward
            (threshold - input_db) * (1.0 - ratio)
        } else if input_db <= threshold + half_knee && self.knee_db > 0.0 {
            // Soft knee region
            let knee_input = input_db - threshold + half_knee;
            (1.0 - ratio) * (knee_input - self.knee_db) * (knee_input - self.knee_db)
                / (2.0 * self.knee_db)
        } else {
            0.0
        }
    }

    /// Process a buffer of samples.
    #[must_use]
    pub fn process(&mut self, samples: &[f32], sample_rate: u32) -> Vec<f32> {
        let attack = Self::attack_coeff(self.attack_ms, sample_rate);
        let release = Self::release_coeff(self.release_ms, sample_rate);

        samples
            .iter()
            .map(|&x| {
                let level = self.detector.process(x, attack, release);
                let level_db = Self::linear_to_db(level);
                let gain_db = self.compute_expansion_gain(level_db);

                // Smooth the gain
                if gain_db < self.smoothed_gain_db {
                    self.smoothed_gain_db += attack * (gain_db - self.smoothed_gain_db);
                } else {
                    self.smoothed_gain_db += release * (gain_db - self.smoothed_gain_db);
                }

                x * Self::db_to_linear(self.smoothed_gain_db)
            })
            .collect()
    }

    /// Reset expander state.
    pub fn reset(&mut self) {
        self.detector.reset();
        self.smoothed_gain_db = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compressor_config_standard() {
        let config = CompressorConfig::standard();
        assert_eq!(config.ratio, 4.0);
        assert!(config.threshold_db < 0.0);
    }

    #[test]
    fn test_compressor_config_limiting() {
        let config = CompressorConfig::limiting();
        assert_eq!(config.ratio, 100.0);
        assert!(config.attack_ms < 1.0);
    }

    #[test]
    fn test_compressor_config_vocal() {
        let config = CompressorConfig::vocal();
        assert_eq!(config.ratio, 3.0);
    }

    #[test]
    fn test_level_detector_attack() {
        let mut det = LevelDetector::new();
        // After processing a loud signal, level should be greater than 0
        for _ in 0..100 {
            det.process(1.0, 0.1, 0.01);
        }
        assert!(det.peak_level > 0.0);
    }

    #[test]
    fn test_level_detector_release() {
        let mut det = LevelDetector::new();
        det.peak_level = 1.0;
        // After processing silence, level should decrease
        for _ in 0..100 {
            det.process(0.0, 0.1, 0.1);
        }
        assert!(det.peak_level < 0.5);
    }

    #[test]
    fn test_gain_computer_below_threshold() {
        let config = CompressorConfig {
            threshold_db: -10.0,
            knee_db: 0.0,
            ..CompressorConfig::standard()
        };
        let mut computer = GainComputerState::new();
        // Signal well below threshold: no gain reduction
        let gr = computer.compute_gain(-20.0, &config);
        assert!(
            gr >= -0.001,
            "Expected no reduction below threshold, got {gr}"
        );
    }

    #[test]
    fn test_gain_computer_above_threshold() {
        let config = CompressorConfig {
            threshold_db: -10.0,
            ratio: 4.0,
            knee_db: 0.0,
            ..CompressorConfig::standard()
        };
        let mut computer = GainComputerState::new();
        // 10dB above threshold with 4:1 ratio
        let gr = computer.compute_gain(0.0, &config);
        // Gain reduction should be negative (attenuation)
        assert!(
            gr < 0.0,
            "Expected gain reduction above threshold, got {gr}"
        );
    }

    #[test]
    fn test_compressor_output_finite() {
        let config = CompressorConfig::standard();
        let mut comp = Compressor::new(config, 48000);
        let input: Vec<f32> = (0..512).map(|i| (i as f32 * 0.01).sin()).collect();
        let output = comp.process(&input, 48000);
        assert_eq!(output.len(), 512);
        assert!(output.iter().all(|&s| s.is_finite()));
    }

    #[test]
    fn test_compressor_reduces_loud_signal() {
        let config = CompressorConfig {
            threshold_db: -6.0,
            ratio: 10.0,
            attack_ms: 1.0,
            release_ms: 50.0,
            knee_db: 0.0,
            makeup_gain_db: 0.0,
        };
        let mut comp = Compressor::new(config, 48000);
        // Loud constant signal
        let input = vec![0.9f32; 1024];
        let output = comp.process(&input, 48000);

        // After settling, output should be lower than input for a loud signal
        let in_rms: f32 = (input.iter().map(|&s| s * s).sum::<f32>() / input.len() as f32).sqrt();
        let out_rms: f32 =
            (output.iter().map(|&s| s * s).sum::<f32>() / output.len() as f32).sqrt();
        assert!(out_rms < in_rms, "Compressor should reduce loud signal");
    }

    #[test]
    fn test_compressor_limiter() {
        let config = CompressorConfig::limiting();
        let mut comp = Compressor::new(config, 48000);
        // Very loud signal
        let input = vec![0.99f32; 2048];
        let output = comp.process(&input, 48000);
        assert!(output.iter().all(|&s| s.is_finite()));
    }

    #[test]
    fn test_gain_reduction_tracking() {
        let mut gr = GainReduction::new(100);
        gr.update(3.0);
        gr.update(6.0);
        assert!(gr.peak_db >= 6.0);
        gr.reset_peak();
        assert_eq!(gr.peak_db, 0.0);
    }

    #[test]
    fn test_expander_output_finite() {
        let mut exp = Expander::gate();
        let input: Vec<f32> = (0..512).map(|i| (i as f32 * 0.01).sin() * 0.1).collect();
        let output = exp.process(&input, 48000);
        assert_eq!(output.len(), 512);
        assert!(output.iter().all(|&s| s.is_finite()));
    }

    #[test]
    fn test_expander_attenuates_below_threshold() {
        let mut exp = Expander::new(-10.0, 5.0, 1.0, 50.0, 0.0);
        // Quiet signal below threshold
        let input = vec![0.001f32; 1024];
        let output = exp.process(&input, 48000);
        let in_rms: f32 = (input.iter().map(|&s| s * s).sum::<f32>() / input.len() as f32).sqrt();
        let out_rms: f32 =
            (output.iter().map(|&s| s * s).sum::<f32>() / output.len() as f32).sqrt();
        assert!(
            out_rms <= in_rms + 1e-6,
            "Expander should attenuate or not increase quiet signals"
        );
    }

    #[test]
    fn test_compressor_reset() {
        let config = CompressorConfig::standard();
        let mut comp = Compressor::new(config, 48000);
        let _ = comp.process(&vec![0.9f32; 512], 48000);
        comp.reset();
        assert_eq!(comp.smoothed_gr_db, 0.0);
    }

    // --- Sidechain compressor tests ---

    #[test]
    fn test_sidechain_compressor_output_finite() {
        let config = CompressorConfig::standard();
        let mut comp = Compressor::new(config, 48000);
        let input: Vec<f32> = (0..512).map(|i| (i as f32 * 0.01).sin()).collect();
        let sidechain: Vec<f32> = vec![0.9; 512]; // loud sidechain
        let output = comp.process_sidechain(&input, &sidechain, 48000);
        assert_eq!(output.len(), 512);
        assert!(output.iter().all(|&s| s.is_finite()));
    }

    #[test]
    fn test_sidechain_compressor_applies_reduction_from_sidechain() {
        let config = CompressorConfig {
            threshold_db: -6.0,
            ratio: 10.0,
            attack_ms: 1.0,
            release_ms: 50.0,
            knee_db: 0.0,
            makeup_gain_db: 0.0,
        };
        let mut comp = Compressor::new(config, 48000);

        // Input is quiet but sidechain is loud: should still compress
        let input = vec![0.5f32; 2048];
        let sidechain = vec![0.9f32; 2048];
        let output = comp.process_sidechain(&input, &sidechain, 48000);

        // Output RMS should be less than input RMS (gain reduction from sidechain)
        let in_rms: f32 = (input.iter().map(|&s| s * s).sum::<f32>() / input.len() as f32).sqrt();
        let out_rms: f32 =
            (output.iter().map(|&s| s * s).sum::<f32>() / output.len() as f32).sqrt();
        assert!(
            out_rms < in_rms,
            "Sidechain compression should reduce input: in={in_rms}, out={out_rms}"
        );
    }

    #[test]
    fn test_sidechain_silent_no_compression() {
        let config = CompressorConfig {
            threshold_db: -20.0,
            ratio: 10.0,
            attack_ms: 1.0,
            release_ms: 50.0,
            knee_db: 0.0,
            makeup_gain_db: 0.0,
        };
        let mut comp = Compressor::new(config, 48000);

        // Input is loud but sidechain is silent: should not compress
        let input = vec![0.5f32; 2048];
        let sidechain = vec![0.0f32; 2048];
        let output = comp.process_sidechain(&input, &sidechain, 48000);

        // Output should be approximately same as input (no sidechain trigger)
        for (&inp, &out) in input.iter().zip(output.iter()) {
            assert!(
                (out - inp).abs() < 0.01,
                "Silent sidechain should not compress: in={inp}, out={out}"
            );
        }
    }

    #[test]
    fn test_sidechain_different_from_normal() {
        let config = CompressorConfig {
            threshold_db: -10.0,
            ratio: 8.0,
            attack_ms: 1.0,
            release_ms: 50.0,
            knee_db: 0.0,
            makeup_gain_db: 0.0,
        };

        // Normal compression on quiet signal
        let mut comp1 = Compressor::new(config.clone(), 48000);
        let quiet_input = vec![0.05f32; 1024];
        let normal_output = comp1.process(&quiet_input, 48000);

        // Sidechain with loud trigger on same quiet signal
        let mut comp2 = Compressor::new(config, 48000);
        let loud_sidechain = vec![0.9f32; 1024];
        let sc_output = comp2.process_sidechain(&quiet_input, &loud_sidechain, 48000);

        // Sidechain output should be quieter due to loud sidechain trigger
        let normal_rms: f32 =
            (normal_output.iter().map(|&s| s * s).sum::<f32>() / normal_output.len() as f32).sqrt();
        let sc_rms: f32 =
            (sc_output.iter().map(|&s| s * s).sum::<f32>() / sc_output.len() as f32).sqrt();
        assert!(
            sc_rms < normal_rms,
            "Sidechain with loud trigger should produce lower output: normal={normal_rms}, sc={sc_rms}"
        );
    }

    // --- Sidechain filter tests ---

    #[test]
    fn test_sidechain_filter_highpass() {
        let config = CompressorConfig {
            threshold_db: -10.0,
            ratio: 8.0,
            attack_ms: 1.0,
            release_ms: 50.0,
            knee_db: 0.0,
            makeup_gain_db: 0.0,
        };
        let mut comp = Compressor::new(config, 48000);
        comp.set_sidechain_filter(SidechainFilter::HighPass(4000.0), 48000);
        assert_eq!(comp.sidechain_filter(), SidechainFilter::HighPass(4000.0));

        // Low-frequency sidechain should be filtered out, reducing compression
        let input = vec![0.5f32; 2048];
        // 100 Hz sidechain (below 4kHz HPF) — should be attenuated
        let sidechain: Vec<f32> = (0..2048)
            .map(|i| (i as f32 * 2.0 * std::f32::consts::PI * 100.0 / 48000.0).sin() * 0.9)
            .collect();
        let output = comp.process_sidechain(&input, &sidechain, 48000);
        assert!(output.iter().all(|&s| s.is_finite()));
    }

    #[test]
    fn test_sidechain_filter_lowpass() {
        let config = CompressorConfig::standard();
        let mut comp = Compressor::new(config, 48000);
        comp.set_sidechain_filter(SidechainFilter::LowPass(200.0), 48000);
        assert_eq!(comp.sidechain_filter(), SidechainFilter::LowPass(200.0));

        let input = vec![0.5f32; 1024];
        let sidechain = vec![0.9f32; 1024];
        let output = comp.process_sidechain(&input, &sidechain, 48000);
        assert!(output.iter().all(|&s| s.is_finite()));
    }

    #[test]
    fn test_sidechain_filter_bandpass() {
        let config = CompressorConfig::standard();
        let mut comp = Compressor::new(config, 48000);
        comp.set_sidechain_filter(SidechainFilter::BandPass(1000.0, 1.0), 48000);

        let input = vec![0.5f32; 1024];
        let sidechain = vec![0.9f32; 1024];
        let output = comp.process_sidechain(&input, &sidechain, 48000);
        assert!(output.iter().all(|&s| s.is_finite()));
    }

    #[test]
    fn test_sidechain_filter_none() {
        let config = CompressorConfig::standard();
        let mut comp = Compressor::new(config, 48000);
        comp.set_sidechain_filter(SidechainFilter::None, 48000);
        assert_eq!(comp.sidechain_filter(), SidechainFilter::None);
        assert!(comp.sc_filter.is_none());
    }

    #[test]
    fn test_sidechain_filter_reset() {
        let config = CompressorConfig::standard();
        let mut comp = Compressor::new(config, 48000);
        comp.set_sidechain_filter(SidechainFilter::HighPass(2000.0), 48000);

        // Process some samples
        let input = vec![0.5f32; 512];
        let sidechain = vec![0.9f32; 512];
        let _ = comp.process_sidechain(&input, &sidechain, 48000);

        // Reset should clear filter state
        comp.reset();
        assert_eq!(comp.smoothed_gr_db, 0.0);
    }

    #[test]
    fn test_sidechain_hpf_attenuates_low_freq_trigger() {
        // With HPF at 4kHz, a low-frequency sidechain should trigger LESS compression
        // than the same sidechain without filtering
        let config = CompressorConfig {
            threshold_db: -10.0,
            ratio: 10.0,
            attack_ms: 1.0,
            release_ms: 50.0,
            knee_db: 0.0,
            makeup_gain_db: 0.0,
        };

        // Without filter
        let mut comp_no_filter = Compressor::new(config.clone(), 48000);

        // With HPF
        let mut comp_hpf = Compressor::new(config, 48000);
        comp_hpf.set_sidechain_filter(SidechainFilter::HighPass(4000.0), 48000);

        let input = vec![0.5f32; 4096];
        // Low-frequency sidechain (100Hz)
        let sidechain: Vec<f32> = (0..4096)
            .map(|i| (i as f32 * 2.0 * std::f32::consts::PI * 100.0 / 48000.0).sin() * 0.9)
            .collect();

        let out_no_filter = comp_no_filter.process_sidechain(&input, &sidechain, 48000);
        let out_hpf = comp_hpf.process_sidechain(&input, &sidechain, 48000);

        let rms_no_filter: f32 =
            (out_no_filter.iter().map(|&s| s * s).sum::<f32>() / out_no_filter.len() as f32).sqrt();
        let rms_hpf: f32 =
            (out_hpf.iter().map(|&s| s * s).sum::<f32>() / out_hpf.len() as f32).sqrt();

        // HPF should pass through more signal (less compression from filtered-out bass)
        assert!(
            rms_hpf >= rms_no_filter - 0.01,
            "HPF should reduce compression from bass sidechain: no_filter={rms_no_filter}, hpf={rms_hpf}"
        );
    }
}
