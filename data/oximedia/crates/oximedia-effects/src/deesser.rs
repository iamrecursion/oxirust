#![allow(dead_code)]

//! De-esser effect for sibilance reduction.
//!
//! A de-esser detects and reduces harsh sibilant frequencies (typically
//! 4 kHz -- 10 kHz) in vocal recordings. It works by band-pass filtering
//! the sidechain signal to isolate sibilance, then applying gain reduction
//! only when that band exceeds a threshold.

/// Operating mode for the de-esser.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeesserMode {
    /// Wideband: reduces gain across the entire signal when sibilance is detected.
    Wideband,
    /// Split-band: only reduces gain in the sibilant frequency band.
    SplitBand,
}

/// Configuration for the de-esser.
#[derive(Debug, Clone)]
pub struct DeesserConfig {
    /// Operating mode.
    pub mode: DeesserMode,
    /// Center frequency of the sibilance band in Hz.
    pub frequency_hz: f32,
    /// Bandwidth in octaves (Q factor derived from this).
    pub bandwidth_octaves: f32,
    /// Threshold in dB above which gain reduction kicks in.
    pub threshold_db: f32,
    /// Maximum gain reduction in dB.
    pub max_reduction_db: f32,
    /// Attack time in ms.
    pub attack_ms: f32,
    /// Release time in ms.
    pub release_ms: f32,
    /// Sample rate in Hz.
    pub sample_rate: f32,
}

impl Default for DeesserConfig {
    fn default() -> Self {
        Self {
            mode: DeesserMode::SplitBand,
            frequency_hz: 6500.0,
            bandwidth_octaves: 1.0,
            threshold_db: -20.0,
            max_reduction_db: -12.0,
            attack_ms: 0.5,
            release_ms: 20.0,
            sample_rate: 48000.0,
        }
    }
}

impl DeesserConfig {
    /// Create a new config with the given sample rate.
    #[must_use]
    pub fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            ..Default::default()
        }
    }

    /// Set mode.
    #[must_use]
    pub fn with_mode(mut self, mode: DeesserMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set center frequency.
    #[must_use]
    pub fn with_frequency(mut self, hz: f32) -> Self {
        self.frequency_hz = hz.clamp(1000.0, 16000.0);
        self
    }

    /// Set bandwidth.
    #[must_use]
    pub fn with_bandwidth(mut self, octaves: f32) -> Self {
        self.bandwidth_octaves = octaves.clamp(0.1, 4.0);
        self
    }

    /// Set threshold.
    #[must_use]
    pub fn with_threshold(mut self, db: f32) -> Self {
        self.threshold_db = db;
        self
    }

    /// Set max reduction.
    #[must_use]
    pub fn with_max_reduction(mut self, db: f32) -> Self {
        self.max_reduction_db = db.min(0.0);
        self
    }

    /// Set attack.
    #[must_use]
    pub fn with_attack(mut self, ms: f32) -> Self {
        self.attack_ms = ms.max(0.01);
        self
    }

    /// Set release.
    #[must_use]
    pub fn with_release(mut self, ms: f32) -> Self {
        self.release_ms = ms.max(1.0);
        self
    }
}

/// Convert dB to linear.
#[allow(clippy::cast_precision_loss)]
fn db_to_linear(db: f32) -> f32 {
    10.0f32.powf(db / 20.0)
}

/// Convert linear to dB.
#[allow(clippy::cast_precision_loss)]
fn linear_to_db(lin: f32) -> f32 {
    if lin <= 1e-10 {
        -200.0
    } else {
        20.0 * lin.log10()
    }
}

/// Compute one-pole smoothing coefficient.
#[allow(clippy::cast_precision_loss)]
fn one_pole_coeff(ms: f32, sr: f32) -> f32 {
    if ms <= 0.0 || sr <= 0.0 {
        return 1.0;
    }
    let n = ms * 0.001 * sr;
    (-1.0f32 / n).exp()
}

/// Second-order band-pass filter (biquad) for sibilance detection.
#[derive(Debug, Clone)]
pub struct BandPassFilter {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl BandPassFilter {
    /// Create a new band-pass filter.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn new(center_hz: f32, bandwidth_octaves: f32, sample_rate: f32) -> Self {
        let w0 = 2.0 * std::f32::consts::PI * center_hz / sample_rate;
        let q = 1.0 / (2.0 * (bandwidth_octaves * (2.0f32.ln()) / 2.0).sinh());
        let alpha = w0.sin() / (2.0 * q);

        let b0 = alpha;
        let b1 = 0.0;
        let b2 = -alpha;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * w0.cos();
        let a2 = 1.0 - alpha;

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    /// Process one sample.
    pub fn process(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }

    /// Reset state.
    pub fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}

/// The de-esser processor.
#[derive(Debug)]
pub struct Deesser {
    config: DeesserConfig,
    /// Band-pass filter to isolate sibilant frequencies.
    bp_filter: BandPassFilter,
    /// Envelope follower for the sidechain.
    envelope: f32,
    /// Attack coefficient.
    attack_coeff: f32,
    /// Release coefficient.
    release_coeff: f32,
    /// Threshold in linear.
    threshold_linear: f32,
    /// Maximum reduction as linear gain.
    max_reduction_linear: f32,
    /// Current gain reduction (1.0 = no reduction).
    gain_reduction: f32,
}

impl Deesser {
    /// Create a new de-esser.
    #[must_use]
    pub fn new(config: DeesserConfig) -> Self {
        let bp_filter = BandPassFilter::new(
            config.frequency_hz,
            config.bandwidth_octaves,
            config.sample_rate,
        );
        let attack_coeff = one_pole_coeff(config.attack_ms, config.sample_rate);
        let release_coeff = one_pole_coeff(config.release_ms, config.sample_rate);
        let threshold_linear = db_to_linear(config.threshold_db);
        let max_reduction_linear = db_to_linear(config.max_reduction_db);

        Self {
            config,
            bp_filter,
            envelope: 0.0,
            attack_coeff,
            release_coeff,
            threshold_linear,
            max_reduction_linear,
            gain_reduction: 1.0,
        }
    }

    /// Process a single sample.
    pub fn process_sample(&mut self, input: f32) -> f32 {
        // Sidechain: band-pass the input to isolate sibilance
        let sidechain = self.bp_filter.process(input);
        let sc_abs = sidechain.abs();

        // Envelope follower
        if sc_abs > self.envelope {
            self.envelope = self.attack_coeff * self.envelope + (1.0 - self.attack_coeff) * sc_abs;
        } else {
            self.envelope =
                self.release_coeff * self.envelope + (1.0 - self.release_coeff) * sc_abs;
        }

        // Compute gain reduction
        let target = if self.envelope > self.threshold_linear {
            let over = self.threshold_linear / self.envelope;
            over.max(self.max_reduction_linear)
        } else {
            1.0
        };

        // Smooth gain
        if target < self.gain_reduction {
            self.gain_reduction =
                self.attack_coeff * self.gain_reduction + (1.0 - self.attack_coeff) * target;
        } else {
            self.gain_reduction =
                self.release_coeff * self.gain_reduction + (1.0 - self.release_coeff) * target;
        }

        match self.config.mode {
            DeesserMode::Wideband => input * self.gain_reduction,
            DeesserMode::SplitBand => {
                // Only reduce the sibilant band
                let non_sibilant = input - sidechain;
                non_sibilant + sidechain * self.gain_reduction
            }
        }
    }

    /// Process a buffer in-place.
    pub fn process(&mut self, buffer: &mut [f32]) {
        for sample in buffer.iter_mut() {
            *sample = self.process_sample(*sample);
        }
    }

    /// Reset internal state.
    pub fn reset(&mut self) {
        self.bp_filter.reset();
        self.envelope = 0.0;
        self.gain_reduction = 1.0;
    }

    /// Get current gain reduction in dB.
    #[must_use]
    pub fn current_reduction_db(&self) -> f32 {
        linear_to_db(self.gain_reduction)
    }

    /// Get current gain reduction (linear).
    #[must_use]
    pub fn current_reduction(&self) -> f32 {
        self.gain_reduction
    }

    /// Get current envelope level.
    #[must_use]
    pub fn envelope_level(&self) -> f32 {
        self.envelope
    }

    /// Update the center frequency.
    pub fn set_frequency(&mut self, hz: f32) {
        self.config.frequency_hz = hz.clamp(1000.0, 16000.0);
        self.bp_filter = BandPassFilter::new(
            self.config.frequency_hz,
            self.config.bandwidth_octaves,
            self.config.sample_rate,
        );
    }

    /// Update the threshold.
    pub fn set_threshold(&mut self, db: f32) {
        self.config.threshold_db = db;
        self.threshold_linear = db_to_linear(db);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_db_to_linear_zero() {
        assert!((db_to_linear(0.0) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_linear_to_db_one() {
        assert!(linear_to_db(1.0).abs() < 1e-5);
    }

    #[test]
    fn test_linear_to_db_tiny() {
        assert_eq!(linear_to_db(0.0), -200.0);
    }

    #[test]
    fn test_one_pole_coeff() {
        let c = one_pole_coeff(10.0, 48000.0);
        assert!(c > 0.0 && c < 1.0);
    }

    #[test]
    fn test_band_pass_filter_creation() {
        let bp = BandPassFilter::new(6500.0, 1.0, 48000.0);
        assert!(bp.b0.is_finite());
        assert!(bp.a1.is_finite());
    }

    #[test]
    fn test_band_pass_filter_dc_rejection() {
        let mut bp = BandPassFilter::new(6500.0, 1.0, 48000.0);
        // Feed DC signal — output should decay to ~0
        for _ in 0..4800 {
            bp.process(1.0);
        }
        let out = bp.process(1.0);
        assert!(out.abs() < 0.01, "Band-pass should reject DC, got {out}");
    }

    #[test]
    fn test_band_pass_reset() {
        let mut bp = BandPassFilter::new(6500.0, 1.0, 48000.0);
        bp.process(1.0);
        bp.reset();
        assert!((bp.x1 - 0.0).abs() < 1e-10);
        assert!((bp.y1 - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_deesser_no_sibilance() {
        let mut deesser = Deesser::new(DeesserConfig::default());
        // Feed low-frequency content (well below sibilance band)
        // Simple low-freq sine approximation
        let mut buf = vec![0.0f32; 4800];
        #[allow(clippy::cast_precision_loss)]
        for (i, s) in buf.iter_mut().enumerate() {
            let phase = 2.0 * std::f32::consts::PI * 200.0 * i as f32 / 48000.0;
            *s = 0.5 * phase.sin();
        }
        deesser.process(&mut buf);
        // Gain reduction should be minimal
        assert!(
            deesser.current_reduction() > 0.9,
            "No sibilance -> minimal reduction, got {}",
            deesser.current_reduction()
        );
    }

    #[test]
    fn test_deesser_with_sibilance() {
        let config = DeesserConfig {
            threshold_db: -40.0,
            max_reduction_db: -12.0,
            frequency_hz: 7000.0,
            attack_ms: 0.5,
            release_ms: 10.0,
            ..DeesserConfig::new(48000.0)
        };
        let mut deesser = Deesser::new(config);
        // Feed high-frequency content at sibilance band
        #[allow(clippy::cast_precision_loss)]
        let buf: Vec<f32> = (0..4800)
            .map(|i| {
                let phase = 2.0 * std::f32::consts::PI * 7000.0 * i as f32 / 48000.0;
                0.8 * phase.sin()
            })
            .collect();
        for &s in &buf {
            deesser.process_sample(s);
        }
        // Should have engaged gain reduction
        assert!(
            deesser.current_reduction() < 0.95,
            "Sibilance present -> reduction expected, got {}",
            deesser.current_reduction()
        );
    }

    #[test]
    fn test_deesser_reset() {
        let mut deesser = Deesser::new(DeesserConfig::default());
        deesser.process_sample(1.0);
        deesser.reset();
        assert!((deesser.envelope_level() - 0.0).abs() < 1e-10);
        assert!((deesser.current_reduction() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_set_frequency() {
        let mut deesser = Deesser::new(DeesserConfig::default());
        deesser.set_frequency(8000.0);
        assert!((deesser.config.frequency_hz - 8000.0).abs() < 1e-5);
    }

    #[test]
    fn test_set_threshold() {
        let mut deesser = Deesser::new(DeesserConfig::default());
        deesser.set_threshold(-15.0);
        assert!((deesser.config.threshold_db - (-15.0)).abs() < 1e-5);
    }

    #[test]
    fn test_config_builder() {
        let cfg = DeesserConfig::new(44100.0)
            .with_mode(DeesserMode::Wideband)
            .with_frequency(5000.0)
            .with_bandwidth(1.5)
            .with_threshold(-25.0)
            .with_max_reduction(-6.0)
            .with_attack(1.0)
            .with_release(30.0);
        assert_eq!(cfg.mode, DeesserMode::Wideband);
        assert!((cfg.frequency_hz - 5000.0).abs() < 1e-5);
        assert!((cfg.bandwidth_octaves - 1.5).abs() < 1e-5);
        assert!((cfg.threshold_db - (-25.0)).abs() < 1e-5);
        assert!((cfg.max_reduction_db - (-6.0)).abs() < 1e-5);
    }

    #[test]
    fn test_wideband_mode() {
        let config = DeesserConfig::new(48000.0).with_mode(DeesserMode::Wideband);
        let mut deesser = Deesser::new(config);
        let out = deesser.process_sample(0.5);
        assert!(out.is_finite());
    }
}
