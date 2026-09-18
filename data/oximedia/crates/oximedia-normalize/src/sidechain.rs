//! Sidechain compression and ducking.
//!
//! Provides sidechain-driven dynamic processing: a control signal (sidechain)
//! drives gain reduction on a main signal. Typical use case is broadcast
//! ducking (music ducks under speech).

/// Sidechain compressor configuration.
#[derive(Clone, Debug)]
pub struct SidechainConfig {
    /// Attack time in milliseconds.
    pub attack_ms: f32,
    /// Release time in milliseconds.
    pub release_ms: f32,
    /// Threshold in dB (above which gain reduction occurs).
    pub threshold_db: f32,
    /// Compression ratio (e.g., 10.0 for 10:1 heavy ducking).
    pub ratio: f32,
    /// Lookahead in milliseconds (0.0 = no lookahead).
    pub lookahead_ms: f32,
}

impl SidechainConfig {
    /// Create a configuration suitable for broadcast ducking.
    ///
    /// Music ducks when speech (sidechain) exceeds -20 dB.
    pub fn ducking() -> Self {
        Self {
            attack_ms: 10.0,
            release_ms: 200.0,
            threshold_db: -20.0,
            ratio: 8.0,
            lookahead_ms: 5.0,
        }
    }

    /// Create a mild ducking configuration.
    pub fn mild() -> Self {
        Self {
            attack_ms: 20.0,
            release_ms: 300.0,
            threshold_db: -18.0,
            ratio: 3.0,
            lookahead_ms: 0.0,
        }
    }

    /// Create an aggressive ducking configuration.
    pub fn aggressive() -> Self {
        Self {
            attack_ms: 5.0,
            release_ms: 100.0,
            threshold_db: -24.0,
            ratio: 20.0,
            lookahead_ms: 10.0,
        }
    }
}

/// Gain reduction meter for monitoring compression amount.
#[derive(Clone, Debug)]
pub struct GainReductionMeter {
    /// Current gain reduction in dB (negative = reduction applied).
    pub current_db: f32,
    /// Peak gain reduction observed (most negative).
    pub peak_db: f32,
}

impl GainReductionMeter {
    /// Create a new gain reduction meter starting at 0 dB reduction.
    pub fn new() -> Self {
        Self {
            current_db: 0.0,
            peak_db: 0.0,
        }
    }

    /// Update the meter with a new gain reduction value.
    ///
    /// `gain_db` should be <= 0.0 (gain reduction).
    pub fn update(&mut self, gain_db: f32) {
        self.current_db = gain_db;
        if gain_db < self.peak_db {
            self.peak_db = gain_db;
        }
    }

    /// Reset peak hold.
    pub fn reset_peak(&mut self) {
        self.peak_db = 0.0;
    }
}

impl Default for GainReductionMeter {
    fn default() -> Self {
        Self::new()
    }
}

/// Sidechain compressor / ducker.
///
/// Uses the sidechain signal level to drive gain reduction on the main signal.
pub struct SidechainCompressor {
    config: SidechainConfig,
    /// Gain reduction meter.
    pub meter: GainReductionMeter,
}

impl SidechainCompressor {
    /// Create a new sidechain compressor.
    pub fn new(config: SidechainConfig) -> Self {
        Self {
            config,
            meter: GainReductionMeter::new(),
        }
    }

    /// Process main audio using sidechain as the control signal.
    ///
    /// Returns the gain-reduced main signal. Both slices must be the same length
    /// and represent mono audio (for multi-channel, pass one channel at a time or
    /// mix down before calling).
    pub fn process(&mut self, main: &[f32], sidechain: &[f32], sample_rate: u32) -> Vec<f32> {
        if main.is_empty() || sidechain.is_empty() || sample_rate == 0 {
            return main.to_vec();
        }

        let len = main.len().min(sidechain.len());
        let sr = f64::from(sample_rate);

        // Time constants: e^(-1 / (sr * t_seconds))
        let attack_samples = (f64::from(self.config.attack_ms) / 1000.0) * sr;
        let release_samples = (f64::from(self.config.release_ms) / 1000.0) * sr;

        let attack_coeff = if attack_samples > 0.0 {
            (-1.0 / attack_samples).exp() as f32
        } else {
            0.0
        };
        let release_coeff = if release_samples > 0.0 {
            (-1.0 / release_samples).exp() as f32
        } else {
            0.0
        };

        let threshold_linear = db_to_linear_f32(self.config.threshold_db);

        // Lookahead: shift sidechain forward in time
        let lookahead_samples = ((f64::from(self.config.lookahead_ms) / 1000.0) * sr) as usize;
        let lookahead_sc: Vec<f32> = if lookahead_samples > 0 {
            let pad = vec![0.0f32; lookahead_samples.min(len)];
            let sc_part = &sidechain[..len.saturating_sub(lookahead_samples)];
            let mut result = pad;
            result.extend_from_slice(sc_part);
            result.truncate(len);
            result
        } else {
            sidechain[..len].to_vec()
        };

        let mut envelope = 0.0_f32;
        let mut output = vec![0.0f32; len];

        for i in 0..len {
            let sc_abs = lookahead_sc[i].abs();

            // Envelope follower with asymmetric attack/release
            if sc_abs > envelope {
                envelope = attack_coeff * envelope + (1.0 - attack_coeff) * sc_abs;
            } else {
                envelope = release_coeff * envelope + (1.0 - release_coeff) * sc_abs;
            }

            // Compute gain reduction
            let gain_db = compute_gain_reduction_db(
                envelope,
                threshold_linear,
                self.config.threshold_db,
                self.config.ratio,
            );

            let gain_linear = db_to_linear_f32(gain_db);
            output[i] = main[i] * gain_linear;

            self.meter.update(gain_db);
        }

        output
    }

    /// Get current configuration.
    pub fn config(&self) -> &SidechainConfig {
        &self.config
    }
}

/// Compute gain reduction in dB given envelope and compressor parameters.
fn compute_gain_reduction_db(
    envelope: f32,
    threshold_linear: f32,
    threshold_db: f32,
    ratio: f32,
) -> f32 {
    if envelope <= threshold_linear || envelope <= 0.0 {
        return 0.0;
    }
    let input_db = 20.0 * envelope.log10();
    let excess_db = input_db - threshold_db;
    // Gain reduction = -excess * (1 - 1/ratio)
    -(excess_db * (1.0 - 1.0 / ratio))
}

/// Convert dB to linear gain (f32).
#[inline]
fn db_to_linear_f32(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ducking_config() {
        let cfg = SidechainConfig::ducking();
        assert!((cfg.attack_ms - 10.0).abs() < 1e-6);
        assert!((cfg.release_ms - 200.0).abs() < 1e-6);
        assert!((cfg.threshold_db - (-20.0)).abs() < 1e-6);
        assert!((cfg.ratio - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_mild_config() {
        let cfg = SidechainConfig::mild();
        assert!((cfg.ratio - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_aggressive_config() {
        let cfg = SidechainConfig::aggressive();
        assert!((cfg.ratio - 20.0).abs() < 1e-6);
        assert!((cfg.attack_ms - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_gain_reduction_meter_new() {
        let m = GainReductionMeter::new();
        assert!((m.current_db - 0.0).abs() < 1e-9);
        assert!((m.peak_db - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_gain_reduction_meter_update() {
        let mut m = GainReductionMeter::new();
        m.update(-6.0);
        assert!((m.current_db - (-6.0)).abs() < 1e-6);
        assert!((m.peak_db - (-6.0)).abs() < 1e-6);

        m.update(-3.0); // Less reduction
        assert!((m.current_db - (-3.0)).abs() < 1e-6);
        assert!((m.peak_db - (-6.0)).abs() < 1e-6); // Peak stays at -6

        m.update(-10.0); // More reduction
        assert!((m.peak_db - (-10.0)).abs() < 1e-6);
    }

    #[test]
    fn test_meter_reset_peak() {
        let mut m = GainReductionMeter::new();
        m.update(-12.0);
        m.reset_peak();
        assert!((m.peak_db - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_process_silence_sidechain() {
        let config = SidechainConfig::ducking();
        let mut comp = SidechainCompressor::new(config);
        let main = vec![1.0f32; 100];
        let sidechain = vec![0.0f32; 100]; // Silent sidechain => no ducking
        let out = comp.process(&main, &sidechain, 48000);
        // With silent sidechain, no gain reduction; output ≈ input
        for s in &out {
            assert!(
                (s - 1.0).abs() < 0.01,
                "Silent sidechain should not duck: got {s}"
            );
        }
    }

    #[test]
    fn test_process_loud_sidechain() {
        let config = SidechainConfig::ducking();
        let mut comp = SidechainCompressor::new(config);
        let main = vec![1.0f32; 4800]; // 100ms at 48kHz
        let sidechain = vec![1.0f32; 4800]; // Full-scale sidechain => max ducking
        let out = comp.process(&main, &sidechain, 48000);
        // After attack settles, output should be significantly reduced
        let last = out[out.len() - 1];
        assert!(
            last < 0.5,
            "Loud sidechain should cause ducking: got {last}"
        );
    }

    #[test]
    fn test_process_empty() {
        let config = SidechainConfig::ducking();
        let mut comp = SidechainCompressor::new(config);
        let out = comp.process(&[], &[], 48000);
        assert!(out.is_empty());
    }

    #[test]
    fn test_gain_reduction_below_threshold() {
        // Signal well below threshold: no reduction
        let thr_linear = db_to_linear_f32(-20.0);
        let gain_db = compute_gain_reduction_db(0.001, thr_linear, -20.0, 8.0);
        assert!(
            (gain_db - 0.0).abs() < 1e-6,
            "Below threshold: no reduction"
        );
    }

    #[test]
    fn test_gain_reduction_above_threshold() {
        // Signal above threshold: should get negative gain reduction
        let thr_linear = db_to_linear_f32(-20.0);
        let input = db_to_linear_f32(-10.0); // 10 dB above threshold
        let gain_db = compute_gain_reduction_db(input, thr_linear, -20.0, 8.0);
        assert!(
            gain_db < 0.0,
            "Above threshold: negative gain reduction expected"
        );
    }

    #[test]
    fn test_db_to_linear() {
        assert!((db_to_linear_f32(0.0) - 1.0).abs() < 1e-6);
        assert!((db_to_linear_f32(20.0) - 10.0).abs() < 1e-4);
        assert!((db_to_linear_f32(-20.0) - 0.1).abs() < 1e-4);
    }

    #[test]
    fn test_config_accessor() {
        let config = SidechainConfig::ducking();
        let comp = SidechainCompressor::new(config.clone());
        assert!((comp.config().attack_ms - config.attack_ms).abs() < 1e-6);
    }
}
