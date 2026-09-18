//! Test signal generator for routing diagnostics.
//!
//! Generates standard test tones (sine, pink noise, sweep) that can be
//! routed through the matrix for alignment, level-setting, and fault-finding.

use std::f64::consts::PI;

/// Type of test signal to generate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalType {
    /// Pure sine wave at a fixed frequency.
    Sine,
    /// Pink noise (1/f spectrum).
    PinkNoise,
    /// White noise (flat spectrum).
    WhiteNoise,
    /// Logarithmic frequency sweep from `start_hz` to `stop_hz`.
    Sweep,
    /// Silence (digital black).
    Silence,
    /// Polarity pulse for phase checking.
    PolarityPulse,
}

/// Configuration for the signal generator.
#[derive(Debug, Clone)]
pub struct GeneratorConfig {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Amplitude in the range 0.0..=1.0 (peak).
    pub amplitude: f64,
    /// Frequency in Hz (used for Sine).
    pub frequency_hz: f64,
    /// Start frequency for sweep.
    pub sweep_start_hz: f64,
    /// Stop frequency for sweep.
    pub sweep_stop_hz: f64,
    /// Duration in seconds for sweep / polarity pulse.
    pub duration_secs: f64,
    /// Signal type.
    pub signal_type: SignalType,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48_000,
            amplitude: 0.5,
            frequency_hz: 1_000.0,
            sweep_start_hz: 20.0,
            sweep_stop_hz: 20_000.0,
            duration_secs: 1.0,
            signal_type: SignalType::Sine,
        }
    }
}

/// A test signal generator that produces audio samples.
#[derive(Debug, Clone)]
pub struct SignalGenerator {
    config: GeneratorConfig,
    /// Current sample index (phase accumulator).
    sample_index: u64,
    /// Pink noise filter state (Voss-McCartney rows).
    pink_rows: [f64; 16],
    /// Pink noise running sum.
    pink_sum: f64,
    /// Pink noise row index.
    pink_index: u32,
    /// Simple LFSR state for noise generation.
    lfsr: u32,
}

impl SignalGenerator {
    /// Creates a new generator with the given configuration.
    pub fn new(config: GeneratorConfig) -> Self {
        Self {
            config,
            sample_index: 0,
            pink_rows: [0.0; 16],
            pink_sum: 0.0,
            pink_index: 0,
            lfsr: 0x1234_5678,
        }
    }

    /// Creates a sine-wave generator at the given frequency.
    pub fn sine(frequency_hz: f64, sample_rate: u32) -> Self {
        Self::new(GeneratorConfig {
            sample_rate,
            frequency_hz,
            signal_type: SignalType::Sine,
            ..GeneratorConfig::default()
        })
    }

    /// Creates a sweep generator.
    pub fn sweep(start_hz: f64, stop_hz: f64, duration_secs: f64, sample_rate: u32) -> Self {
        Self::new(GeneratorConfig {
            sample_rate,
            sweep_start_hz: start_hz,
            sweep_stop_hz: stop_hz,
            duration_secs,
            signal_type: SignalType::Sweep,
            ..GeneratorConfig::default()
        })
    }

    /// Creates a pink noise generator.
    pub fn pink_noise(sample_rate: u32) -> Self {
        Self::new(GeneratorConfig {
            sample_rate,
            signal_type: SignalType::PinkNoise,
            ..GeneratorConfig::default()
        })
    }

    /// Creates a white noise generator.
    pub fn white_noise(sample_rate: u32) -> Self {
        Self::new(GeneratorConfig {
            sample_rate,
            signal_type: SignalType::WhiteNoise,
            ..GeneratorConfig::default()
        })
    }

    /// Returns the current configuration.
    pub fn config(&self) -> &GeneratorConfig {
        &self.config
    }

    /// Sets the amplitude (0.0..=1.0).
    pub fn set_amplitude(&mut self, amplitude: f64) {
        self.config.amplitude = amplitude.clamp(0.0, 1.0);
    }

    /// Sets the frequency for sine wave generation.
    pub fn set_frequency(&mut self, hz: f64) {
        self.config.frequency_hz = hz;
    }

    /// Resets the generator phase to the beginning.
    pub fn reset(&mut self) {
        self.sample_index = 0;
        self.pink_rows = [0.0; 16];
        self.pink_sum = 0.0;
        self.pink_index = 0;
        self.lfsr = 0x1234_5678;
    }

    /// Returns the current sample index.
    pub fn sample_index(&self) -> u64 {
        self.sample_index
    }

    /// Generates the next sample as f64.
    pub fn next_sample(&mut self) -> f64 {
        let sample = match self.config.signal_type {
            SignalType::Sine => self.gen_sine(),
            SignalType::PinkNoise => self.gen_pink_noise(),
            SignalType::WhiteNoise => self.gen_white_noise(),
            SignalType::Sweep => self.gen_sweep(),
            SignalType::Silence => 0.0,
            SignalType::PolarityPulse => self.gen_polarity_pulse(),
        };
        self.sample_index += 1;
        sample * self.config.amplitude
    }

    /// Fills a buffer with generated samples.
    pub fn fill_buffer(&mut self, buffer: &mut [f64]) {
        for sample in buffer.iter_mut() {
            *sample = self.next_sample();
        }
    }

    /// Generates a block of samples as a Vec.
    pub fn generate_block(&mut self, count: usize) -> Vec<f64> {
        let mut buf = vec![0.0; count];
        self.fill_buffer(&mut buf);
        buf
    }

    /// Generates a block of f32 samples (for routing through the matrix).
    pub fn generate_block_f32(&mut self, count: usize) -> Vec<f32> {
        (0..count).map(|_| self.next_sample() as f32).collect()
    }

    // --- Internal generators ---

    fn gen_sine(&self) -> f64 {
        let phase = 2.0 * PI * self.config.frequency_hz * self.sample_index as f64
            / self.config.sample_rate as f64;
        phase.sin()
    }

    fn gen_sweep(&self) -> f64 {
        let total_samples = (self.config.duration_secs * self.config.sample_rate as f64) as u64;
        if total_samples == 0 {
            return 0.0;
        }
        let t = (self.sample_index % total_samples) as f64 / self.config.sample_rate as f64;
        let f0 = self.config.sweep_start_hz;
        let f1 = self.config.sweep_stop_hz;
        let dur = self.config.duration_secs;

        // Logarithmic sweep: f(t) = f0 * (f1/f0)^(t/T)
        let ratio = f1 / f0;
        if ratio <= 0.0 || dur <= 0.0 {
            return 0.0;
        }
        let log_ratio = ratio.ln();
        let phase = 2.0 * PI * f0 * dur / log_ratio * ((log_ratio * t / dur).exp() - 1.0);
        phase.sin()
    }

    fn gen_white_noise(&mut self) -> f64 {
        self.lfsr_next_f64()
    }

    fn gen_pink_noise(&mut self) -> f64 {
        // Voss-McCartney algorithm: update one row per sample based on
        // trailing zeros of the index.
        let tz = self.pink_index.trailing_zeros().min(15) as usize;
        self.pink_sum -= self.pink_rows[tz];
        let new_val = self.lfsr_next_f64();
        self.pink_rows[tz] = new_val;
        self.pink_sum += new_val;
        self.pink_index = self.pink_index.wrapping_add(1);

        // Normalize roughly (16 rows contribute, so divide by ~16)
        self.pink_sum / 16.0
    }

    fn gen_polarity_pulse(&self) -> f64 {
        // A single positive-going pulse at the start, then silence.
        let pulse_len = (self.config.sample_rate as u64) / 1000; // 1 ms
        if self.sample_index < pulse_len {
            1.0
        } else {
            0.0
        }
    }

    /// Simple LFSR-based pseudo-random in [-1, 1].
    fn lfsr_next_f64(&mut self) -> f64 {
        // Galois LFSR with taps at bits 31, 21, 1, 0
        let bit = self.lfsr & 1;
        self.lfsr >>= 1;
        if bit == 1 {
            self.lfsr ^= 0xB400_0000;
        }
        // Map u32 to [-1.0, 1.0]
        (self.lfsr as f64 / (u32::MAX as f64 / 2.0)) - 1.0
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sine_generator_basic() {
        let mut gen = SignalGenerator::sine(1000.0, 48000);
        gen.set_amplitude(1.0);
        let block = gen.generate_block(48000);
        assert_eq!(block.len(), 48000);
        // At sample 0 the sine should be ~0
        assert!(block[0].abs() < 1e-10);
    }

    #[test]
    fn test_sine_amplitude() {
        let mut gen = SignalGenerator::sine(1000.0, 48000);
        gen.set_amplitude(0.5);
        let block = gen.generate_block(48000);
        // All samples should be <= 0.5 in magnitude
        for &s in &block {
            assert!(s.abs() <= 0.5 + 1e-10);
        }
    }

    #[test]
    fn test_sine_quarter_period() {
        // At quarter period of a 1 Hz sine at SR=4, sample 1 should be peak
        let mut gen = SignalGenerator::sine(1.0, 4);
        gen.set_amplitude(1.0);
        let block = gen.generate_block(4);
        // sample 0=0, sample 1=1, sample 2≈0, sample 3=-1
        assert!(block[0].abs() < 1e-10);
        assert!((block[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_silence_generator() {
        let mut gen = SignalGenerator::new(GeneratorConfig {
            signal_type: SignalType::Silence,
            ..GeneratorConfig::default()
        });
        let block = gen.generate_block(100);
        for &s in &block {
            assert!(s.abs() < 1e-15);
        }
    }

    #[test]
    fn test_white_noise_not_silent() {
        let mut gen = SignalGenerator::white_noise(48000);
        gen.set_amplitude(1.0);
        let block = gen.generate_block(1000);
        let energy: f64 = block.iter().map(|s| s * s).sum();
        assert!(energy > 0.0);
    }

    #[test]
    fn test_pink_noise_not_silent() {
        let mut gen = SignalGenerator::pink_noise(48000);
        gen.set_amplitude(1.0);
        let block = gen.generate_block(1000);
        let energy: f64 = block.iter().map(|s| s * s).sum();
        assert!(energy > 0.0);
    }

    #[test]
    fn test_sweep_monotonic_phase() {
        let mut gen = SignalGenerator::sweep(20.0, 20000.0, 1.0, 48000);
        gen.set_amplitude(1.0);
        let block = gen.generate_block(480);
        // Just verify no NaN or inf
        for &s in &block {
            assert!(s.is_finite());
            assert!(s.abs() <= 1.0 + 1e-10);
        }
    }

    #[test]
    fn test_polarity_pulse() {
        let mut gen = SignalGenerator::new(GeneratorConfig {
            signal_type: SignalType::PolarityPulse,
            sample_rate: 48000,
            amplitude: 1.0,
            ..GeneratorConfig::default()
        });
        let block = gen.generate_block(100);
        // First ~48 samples should be 1.0 (1 ms at 48 kHz)
        assert!((block[0] - 1.0).abs() < 1e-10);
        assert!((block[47] - 1.0).abs() < 1e-10);
        // After 48 samples, silence
        assert!(block[49].abs() < 1e-10);
    }

    #[test]
    fn test_reset() {
        let mut gen = SignalGenerator::sine(1000.0, 48000);
        gen.set_amplitude(1.0);
        let a = gen.next_sample();
        gen.reset();
        let b = gen.next_sample();
        assert!((a - b).abs() < 1e-15);
    }

    #[test]
    fn test_generate_block_f32() {
        let mut gen = SignalGenerator::sine(1000.0, 48000);
        let block = gen.generate_block_f32(100);
        assert_eq!(block.len(), 100);
    }

    #[test]
    fn test_fill_buffer() {
        let mut gen = SignalGenerator::sine(440.0, 48000);
        let mut buf = [0.0_f64; 256];
        gen.fill_buffer(&mut buf);
        // Should have non-zero energy
        let energy: f64 = buf.iter().map(|s| s * s).sum();
        assert!(energy > 0.0);
    }

    #[test]
    fn test_sample_index_advances() {
        let mut gen = SignalGenerator::sine(1000.0, 48000);
        assert_eq!(gen.sample_index(), 0);
        gen.next_sample();
        assert_eq!(gen.sample_index(), 1);
        gen.generate_block(99);
        assert_eq!(gen.sample_index(), 100);
    }

    #[test]
    fn test_config_access() {
        let gen = SignalGenerator::sine(440.0, 44100);
        assert_eq!(gen.config().sample_rate, 44100);
        assert!((gen.config().frequency_hz - 440.0).abs() < 1e-10);
    }

    #[test]
    fn test_set_frequency() {
        let mut gen = SignalGenerator::sine(440.0, 48000);
        gen.set_frequency(880.0);
        assert!((gen.config().frequency_hz - 880.0).abs() < 1e-10);
    }

    #[test]
    fn test_amplitude_clamped() {
        let mut gen = SignalGenerator::sine(440.0, 48000);
        gen.set_amplitude(2.0);
        assert!((gen.config().amplitude - 1.0).abs() < 1e-10);
        gen.set_amplitude(-0.5);
        assert!(gen.config().amplitude.abs() < 1e-10);
    }

    #[test]
    fn test_default_config() {
        let cfg = GeneratorConfig::default();
        assert_eq!(cfg.sample_rate, 48_000);
        assert!((cfg.amplitude - 0.5).abs() < 1e-10);
        assert!((cfg.frequency_hz - 1_000.0).abs() < 1e-10);
        assert_eq!(cfg.signal_type, SignalType::Sine);
    }
}
