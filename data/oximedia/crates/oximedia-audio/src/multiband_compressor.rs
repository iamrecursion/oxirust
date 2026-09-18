//! Multi-band dynamic range compressor.
//!
//! Splits the audio signal into multiple frequency bands using a Linkwitz-Riley
//! crossover network, applies independent compression to each band, and sums
//! the results. This allows frequency-selective dynamics control without the
//! pumping artifacts of wideband compression.
//!
//! # Architecture
//!
//! ```text
//!              ┌──► Band 1 (low)  ──► Compressor 1 ──┐
//! Input ──► Crossover ──► Band 2 (mid)  ──► Compressor 2 ──► Sum ──► Output
//!              └──► Band 3 (high) ──► Compressor 3 ──┘
//! ```
//!
//! The crossover uses cascaded second-order Butterworth filters (Linkwitz-Riley
//! 4th order) for flat magnitude response at the crossover frequencies.

#![forbid(unsafe_code)]

use std::f32::consts::PI;

/// Maximum number of bands supported.
pub const MAX_BANDS: usize = 5;

/// Minimum number of bands.
pub const MIN_BANDS: usize = 2;

// ---------------------------------------------------------------------------
// Biquad filter (f32, used internally for crossover)
// ---------------------------------------------------------------------------

/// Second-order IIR biquad filter for crossover network.
#[derive(Clone, Debug)]
struct CrossoverBiquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: f32,
    z2: f32,
}

impl CrossoverBiquad {
    fn lowpass(freq: f32, q: f32, sample_rate: f32) -> Self {
        let w0 = 2.0 * PI * freq / sample_rate;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / (2.0 * q);

        let b0 = (1.0 - cos_w0) / 2.0;
        let b1 = 1.0 - cos_w0;
        let b2 = (1.0 - cos_w0) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    fn highpass(freq: f32, q: f32, sample_rate: f32) -> Self {
        let w0 = 2.0 * PI * freq / sample_rate;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / (2.0 * q);

        let b0 = (1.0 + cos_w0) / 2.0;
        let b1 = -(1.0 + cos_w0);
        let b2 = (1.0 + cos_w0) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    fn process(&mut self, input: f32) -> f32 {
        let output = self.b0 * input + self.z1;
        self.z1 = self.b1 * input - self.a1 * output + self.z2;
        self.z2 = self.b2 * input - self.a2 * output;
        output
    }

    fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Linkwitz-Riley crossover (4th-order = two cascaded Butterworth 2nd-order)
// ---------------------------------------------------------------------------

/// A single crossover point that splits a signal into low and high bands.
#[derive(Clone, Debug)]
struct CrossoverPoint {
    /// Two cascaded lowpass filters (LR4).
    lp: [CrossoverBiquad; 2],
    /// Two cascaded highpass filters (LR4).
    hp: [CrossoverBiquad; 2],
    /// Crossover frequency in Hz.
    frequency: f32,
}

impl CrossoverPoint {
    fn new(frequency: f32, sample_rate: f32) -> Self {
        let q = std::f32::consts::FRAC_1_SQRT_2; // Butterworth Q
        Self {
            lp: [
                CrossoverBiquad::lowpass(frequency, q, sample_rate),
                CrossoverBiquad::lowpass(frequency, q, sample_rate),
            ],
            hp: [
                CrossoverBiquad::highpass(frequency, q, sample_rate),
                CrossoverBiquad::highpass(frequency, q, sample_rate),
            ],
            frequency,
        }
    }

    /// Split an input sample into low and high components.
    fn split(&mut self, input: f32) -> (f32, f32) {
        let mut low = input;
        for lp in &mut self.lp {
            low = lp.process(low);
        }

        let mut high = input;
        for hp in &mut self.hp {
            high = hp.process(high);
        }

        (low, high)
    }

    fn reset(&mut self) {
        for lp in &mut self.lp {
            lp.reset();
        }
        for hp in &mut self.hp {
            hp.reset();
        }
    }
}

// ---------------------------------------------------------------------------
// Per-band compressor
// ---------------------------------------------------------------------------

/// Compression parameters for a single frequency band.
#[derive(Clone, Debug)]
pub struct BandCompressorConfig {
    /// Threshold in dBFS above which compression begins.
    pub threshold_db: f32,
    /// Compression ratio (e.g., 4.0 = 4:1).
    pub ratio: f32,
    /// Attack time in seconds.
    pub attack_secs: f32,
    /// Release time in seconds.
    pub release_secs: f32,
    /// Make-up gain in dB applied after compression.
    pub makeup_gain_db: f32,
    /// Whether this band is soloed (for monitoring).
    pub solo: bool,
    /// Whether this band is muted.
    pub mute: bool,
}

impl Default for BandCompressorConfig {
    fn default() -> Self {
        Self {
            threshold_db: -20.0,
            ratio: 4.0,
            attack_secs: 0.01,
            release_secs: 0.1,
            makeup_gain_db: 0.0,
            solo: false,
            mute: false,
        }
    }
}

/// Internal state for a single band's compressor.
#[derive(Clone, Debug)]
struct BandCompressorState {
    envelope: f32,
    attack_coeff: f32,
    release_coeff: f32,
    last_gain_reduction_db: f32,
}

impl BandCompressorState {
    fn new(config: &BandCompressorConfig, sample_rate: f32) -> Self {
        Self {
            envelope: 0.0,
            attack_coeff: time_to_coeff(config.attack_secs, sample_rate),
            release_coeff: time_to_coeff(config.release_secs, sample_rate),
            last_gain_reduction_db: 0.0,
        }
    }

    fn process(&mut self, input: f32, config: &BandCompressorConfig) -> f32 {
        let abs_input = input.abs();

        // Envelope follower
        if abs_input > self.envelope {
            self.envelope =
                self.attack_coeff * self.envelope + (1.0 - self.attack_coeff) * abs_input;
        } else {
            self.envelope =
                self.release_coeff * self.envelope + (1.0 - self.release_coeff) * abs_input;
        }

        // Level in dB
        let level_db = if self.envelope > 1e-10 {
            20.0 * self.envelope.log10()
        } else {
            -120.0
        };

        // Gain reduction
        let gr_db = if level_db > config.threshold_db {
            (level_db - config.threshold_db) * (1.0 - 1.0 / config.ratio)
        } else {
            0.0
        };
        self.last_gain_reduction_db = gr_db;

        // Apply gain
        let total_gain_db = -gr_db + config.makeup_gain_db;
        let gain_linear = 10.0_f32.powf(total_gain_db / 20.0);

        input * gain_linear
    }

    fn reset(&mut self) {
        self.envelope = 0.0;
        self.last_gain_reduction_db = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Multi-band compressor configuration
// ---------------------------------------------------------------------------

/// Configuration for the multi-band compressor.
#[derive(Clone, Debug)]
pub struct MultibandCompressorConfig {
    /// Crossover frequencies in Hz (sorted ascending).
    /// The number of bands = crossover_frequencies.len() + 1.
    pub crossover_frequencies: Vec<f32>,
    /// Per-band compression settings.
    /// Must have exactly crossover_frequencies.len() + 1 entries.
    pub band_configs: Vec<BandCompressorConfig>,
    /// Output gain in dB.
    pub output_gain_db: f32,
    /// Sample rate in Hz.
    pub sample_rate: f32,
}

impl Default for MultibandCompressorConfig {
    fn default() -> Self {
        // Default 3-band: 200 Hz / 2000 Hz crossover
        let crossover_frequencies = vec![200.0, 2000.0];
        let band_configs = vec![
            BandCompressorConfig {
                threshold_db: -18.0,
                ratio: 3.0,
                attack_secs: 0.02,
                release_secs: 0.15,
                makeup_gain_db: 0.0,
                ..Default::default()
            },
            BandCompressorConfig::default(),
            BandCompressorConfig {
                threshold_db: -24.0,
                ratio: 2.5,
                attack_secs: 0.005,
                release_secs: 0.08,
                makeup_gain_db: 0.0,
                ..Default::default()
            },
        ];

        Self {
            crossover_frequencies,
            band_configs,
            output_gain_db: 0.0,
            sample_rate: 48_000.0,
        }
    }
}

impl MultibandCompressorConfig {
    /// Create a configuration with the given crossover frequencies.
    ///
    /// Band configs are initialized to defaults.
    ///
    /// # Arguments
    ///
    /// * `crossover_frequencies` - Crossover frequencies in Hz (sorted ascending)
    /// * `sample_rate` - Sample rate in Hz
    pub fn new(mut crossover_frequencies: Vec<f32>, sample_rate: f32) -> Self {
        crossover_frequencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let num_bands = crossover_frequencies.len() + 1;
        let band_configs = (0..num_bands)
            .map(|_| BandCompressorConfig::default())
            .collect();

        Self {
            crossover_frequencies,
            band_configs,
            output_gain_db: 0.0,
            sample_rate,
        }
    }

    /// Return the number of bands.
    pub fn num_bands(&self) -> usize {
        self.crossover_frequencies.len() + 1
    }

    /// Create a mastering preset (4-band).
    pub fn mastering(sample_rate: f32) -> Self {
        let crossover_frequencies = vec![100.0, 1000.0, 8000.0];
        let band_configs = vec![
            BandCompressorConfig {
                threshold_db: -16.0,
                ratio: 3.0,
                attack_secs: 0.03,
                release_secs: 0.2,
                makeup_gain_db: 1.0,
                ..Default::default()
            },
            BandCompressorConfig {
                threshold_db: -20.0,
                ratio: 2.5,
                attack_secs: 0.01,
                release_secs: 0.1,
                makeup_gain_db: 0.5,
                ..Default::default()
            },
            BandCompressorConfig {
                threshold_db: -22.0,
                ratio: 2.0,
                attack_secs: 0.008,
                release_secs: 0.08,
                makeup_gain_db: 0.0,
                ..Default::default()
            },
            BandCompressorConfig {
                threshold_db: -26.0,
                ratio: 1.5,
                attack_secs: 0.005,
                release_secs: 0.06,
                makeup_gain_db: -0.5,
                ..Default::default()
            },
        ];

        Self {
            crossover_frequencies,
            band_configs,
            output_gain_db: 0.0,
            sample_rate,
        }
    }

    /// Create a broadcast preset (3-band, aggressive).
    pub fn broadcast(sample_rate: f32) -> Self {
        let crossover_frequencies = vec![200.0, 3000.0];
        let band_configs = vec![
            BandCompressorConfig {
                threshold_db: -14.0,
                ratio: 5.0,
                attack_secs: 0.015,
                release_secs: 0.12,
                makeup_gain_db: 2.0,
                ..Default::default()
            },
            BandCompressorConfig {
                threshold_db: -16.0,
                ratio: 4.0,
                attack_secs: 0.01,
                release_secs: 0.1,
                makeup_gain_db: 1.5,
                ..Default::default()
            },
            BandCompressorConfig {
                threshold_db: -20.0,
                ratio: 3.0,
                attack_secs: 0.005,
                release_secs: 0.08,
                makeup_gain_db: 1.0,
                ..Default::default()
            },
        ];

        Self {
            crossover_frequencies,
            band_configs,
            output_gain_db: 0.0,
            sample_rate,
        }
    }
}

// ---------------------------------------------------------------------------
// Multi-band compressor processor
// ---------------------------------------------------------------------------

/// Multi-band dynamic range compressor.
///
/// Splits the input signal into frequency bands via Linkwitz-Riley crossover
/// filters, compresses each band independently, and sums them.
pub struct MultibandCompressor {
    config: MultibandCompressorConfig,
    crossovers: Vec<CrossoverPoint>,
    band_states: Vec<BandCompressorState>,
    output_gain_linear: f32,
    /// Temporary per-band sample storage.
    band_samples: Vec<f32>,
}

impl MultibandCompressor {
    /// Create a new multi-band compressor.
    ///
    /// # Arguments
    ///
    /// * `config` - Multi-band compressor configuration
    pub fn new(config: MultibandCompressorConfig) -> Self {
        let crossovers: Vec<CrossoverPoint> = config
            .crossover_frequencies
            .iter()
            .map(|&freq| CrossoverPoint::new(freq, config.sample_rate))
            .collect();

        let num_bands = config.num_bands();
        let band_states: Vec<BandCompressorState> = config
            .band_configs
            .iter()
            .take(num_bands)
            .map(|bc| BandCompressorState::new(bc, config.sample_rate))
            .collect();

        let output_gain_linear = 10.0_f32.powf(config.output_gain_db / 20.0);
        let band_samples = vec![0.0; num_bands];

        Self {
            config,
            crossovers,
            band_states,
            output_gain_linear,
            band_samples,
        }
    }

    /// Process a single sample through the multi-band compressor.
    pub fn process_sample(&mut self, input: f32) -> f32 {
        let num_bands = self.config.num_bands();

        // Split into bands using cascaded crossover filters
        self.split_into_bands(input);

        // Compress each band independently
        let mut output = 0.0_f32;
        for i in 0..num_bands {
            if i >= self.band_states.len() || i >= self.config.band_configs.len() {
                continue;
            }

            let band_config = &self.config.band_configs[i];
            if band_config.mute {
                continue;
            }

            let compressed = self.band_states[i].process(self.band_samples[i], band_config);
            output += compressed;
        }

        output * self.output_gain_linear
    }

    /// Split the input sample into frequency bands.
    fn split_into_bands(&mut self, input: f32) {
        let num_crossovers = self.crossovers.len();

        if num_crossovers == 0 {
            // Single band: no crossover
            if !self.band_samples.is_empty() {
                self.band_samples[0] = input;
            }
            return;
        }

        // First crossover splits into band 0 (low) and remainder (high)
        let (low, mut remainder) = self.crossovers[0].split(input);
        self.band_samples[0] = low;

        // Each subsequent crossover splits the remainder
        for i in 1..num_crossovers {
            let (low_part, high_part) = self.crossovers[i].split(remainder);
            self.band_samples[i] = low_part;
            remainder = high_part;
        }

        // The last band gets the final remainder
        self.band_samples[num_crossovers] = remainder;
    }

    /// Process a buffer of samples in-place.
    pub fn process_buffer(&mut self, samples: &mut [f32]) {
        for s in samples.iter_mut() {
            *s = self.process_sample(*s);
        }
    }

    /// Get the gain reduction in dB for a specific band.
    ///
    /// # Arguments
    ///
    /// * `band` - Band index (0-based)
    pub fn band_gain_reduction_db(&self, band: usize) -> f32 {
        self.band_states
            .get(band)
            .map_or(0.0, |s| s.last_gain_reduction_db)
    }

    /// Get gain reduction for all bands.
    pub fn all_gain_reductions_db(&self) -> Vec<f32> {
        self.band_states
            .iter()
            .map(|s| s.last_gain_reduction_db)
            .collect()
    }

    /// Get the number of bands.
    pub fn num_bands(&self) -> usize {
        self.config.num_bands()
    }

    /// Get the crossover frequencies.
    pub fn crossover_frequencies(&self) -> &[f32] {
        &self.config.crossover_frequencies
    }

    /// Update the compression settings for a specific band.
    ///
    /// # Arguments
    ///
    /// * `band` - Band index
    /// * `config` - New compression settings
    pub fn set_band_config(&mut self, band: usize, band_config: BandCompressorConfig) {
        if band < self.config.band_configs.len() {
            let sample_rate = self.config.sample_rate;
            self.band_states[band] = BandCompressorState::new(&band_config, sample_rate);
            self.config.band_configs[band] = band_config;
        }
    }

    /// Set the output gain in dB.
    pub fn set_output_gain_db(&mut self, gain_db: f32) {
        self.config.output_gain_db = gain_db;
        self.output_gain_linear = 10.0_f32.powf(gain_db / 20.0);
    }

    /// Reset all internal state.
    pub fn reset(&mut self) {
        for xo in &mut self.crossovers {
            xo.reset();
        }
        for bs in &mut self.band_states {
            bs.reset();
        }
        self.band_samples.fill(0.0);
    }
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn time_to_coeff(time_secs: f32, sample_rate: f32) -> f32 {
    if time_secs <= 0.0 || sample_rate <= 0.0 {
        return 0.0;
    }
    (-1.0_f32 / (time_secs * sample_rate)).exp()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_default() -> MultibandCompressor {
        MultibandCompressor::new(MultibandCompressorConfig::default())
    }

    #[test]
    fn test_creation_default() {
        let mbc = make_default();
        assert_eq!(mbc.num_bands(), 3);
        assert_eq!(mbc.crossover_frequencies().len(), 2);
    }

    #[test]
    fn test_creation_mastering() {
        let config = MultibandCompressorConfig::mastering(48_000.0);
        let mbc = MultibandCompressor::new(config);
        assert_eq!(mbc.num_bands(), 4);
    }

    #[test]
    fn test_creation_broadcast() {
        let config = MultibandCompressorConfig::broadcast(48_000.0);
        let mbc = MultibandCompressor::new(config);
        assert_eq!(mbc.num_bands(), 3);
    }

    #[test]
    fn test_silence_passthrough() {
        let mut mbc = make_default();
        let out = mbc.process_sample(0.0);
        assert_eq!(out, 0.0);
    }

    #[test]
    fn test_output_is_finite() {
        let mut mbc = make_default();
        for _ in 0..1000 {
            let out = mbc.process_sample(0.5);
            assert!(out.is_finite(), "output must be finite");
        }
    }

    #[test]
    fn test_process_buffer() {
        let mut mbc = make_default();
        let mut buf = vec![0.3_f32; 2000];
        mbc.process_buffer(&mut buf);
        for s in &buf {
            assert!(s.is_finite());
        }
    }

    #[test]
    fn test_gain_reduction_increases_on_loud_signal() {
        let mut mbc = make_default();
        // Push loud signal
        for _ in 0..5000 {
            mbc.process_sample(0.9);
        }
        // At least one band should show gain reduction
        let gr = mbc.all_gain_reductions_db();
        let total: f32 = gr.iter().sum();
        assert!(total > 0.0, "some gain reduction expected on loud signal");
    }

    #[test]
    fn test_mute_band_silences_output() {
        let mut config = MultibandCompressorConfig::default();
        // Mute all bands
        for bc in &mut config.band_configs {
            bc.mute = true;
        }
        let mut mbc = MultibandCompressor::new(config);

        // Process loud signal — output should be 0 since all bands muted
        for _ in 0..1000 {
            let out = mbc.process_sample(0.8);
            assert!(out.abs() < 1e-10, "muted bands should produce silence");
        }
    }

    #[test]
    fn test_set_band_config() {
        let mut mbc = make_default();
        let new_cfg = BandCompressorConfig {
            threshold_db: -10.0,
            ratio: 8.0,
            ..Default::default()
        };
        mbc.set_band_config(0, new_cfg);
        assert_eq!(mbc.config.band_configs[0].threshold_db, -10.0);
        assert_eq!(mbc.config.band_configs[0].ratio, 8.0);
    }

    #[test]
    fn test_set_output_gain() {
        let mut mbc = make_default();
        mbc.set_output_gain_db(6.0);
        assert!((mbc.output_gain_linear - 10.0_f32.powf(6.0 / 20.0)).abs() < 1e-5);
    }

    #[test]
    fn test_reset() {
        let mut mbc = make_default();
        for _ in 0..1000 {
            mbc.process_sample(0.7);
        }
        mbc.reset();
        for state in &mbc.band_states {
            assert_eq!(state.envelope, 0.0);
            assert_eq!(state.last_gain_reduction_db, 0.0);
        }
    }

    #[test]
    fn test_crossover_band_split_sums_to_input() {
        // With a single crossover point, the low + high should approximately
        // reconstruct the original (within the filter's group delay).
        let mut xo = CrossoverPoint::new(1000.0, 48_000.0);

        // Warm up the filter
        for _ in 0..5000 {
            xo.split(0.5);
        }

        // Check that low + high ≈ input for a steady-state DC signal
        let (low, high) = xo.split(0.5);
        let sum = low + high;
        assert!(
            (sum - 0.5).abs() < 0.05,
            "crossover should reconstruct input, got {sum}"
        );
    }

    #[test]
    fn test_crossover_biquad_lowpass() {
        let mut lp = CrossoverBiquad::lowpass(1000.0, 0.707, 48_000.0);
        // DC signal should pass through
        for _ in 0..5000 {
            lp.process(1.0);
        }
        let out = lp.process(1.0);
        assert!(
            (out - 1.0).abs() < 0.02,
            "lowpass should pass DC, got {out}"
        );
    }

    #[test]
    fn test_crossover_biquad_highpass() {
        let mut hp = CrossoverBiquad::highpass(1000.0, 0.707, 48_000.0);
        // DC signal should be blocked
        for _ in 0..5000 {
            hp.process(1.0);
        }
        let out = hp.process(1.0);
        assert!(out.abs() < 0.02, "highpass should block DC, got {out}");
    }

    #[test]
    fn test_time_to_coeff_edge_cases() {
        assert_eq!(time_to_coeff(0.0, 48_000.0), 0.0);
        assert_eq!(time_to_coeff(0.01, 0.0), 0.0);
        assert_eq!(time_to_coeff(-1.0, 48_000.0), 0.0);
        let c = time_to_coeff(0.01, 48_000.0);
        assert!(c > 0.0 && c < 1.0);
    }

    #[test]
    fn test_two_band_config() {
        let config = MultibandCompressorConfig::new(vec![500.0], 48_000.0);
        let mbc = MultibandCompressor::new(config);
        assert_eq!(mbc.num_bands(), 2);
    }

    #[test]
    fn test_five_band_config() {
        let config = MultibandCompressorConfig::new(vec![100.0, 500.0, 2000.0, 8000.0], 48_000.0);
        let mut mbc = MultibandCompressor::new(config);
        assert_eq!(mbc.num_bands(), 5);
        for _ in 0..1000 {
            let out = mbc.process_sample(0.4);
            assert!(out.is_finite());
        }
    }

    #[test]
    fn test_band_gain_reduction_valid_index() {
        let mut mbc = make_default();
        for _ in 0..2000 {
            mbc.process_sample(0.8);
        }
        let gr = mbc.band_gain_reduction_db(0);
        assert!(gr >= 0.0);
    }

    #[test]
    fn test_band_gain_reduction_invalid_index() {
        let mbc = make_default();
        assert_eq!(mbc.band_gain_reduction_db(999), 0.0);
    }

    #[test]
    fn test_compression_reduces_loud_signal_level() {
        let config = MultibandCompressorConfig {
            crossover_frequencies: vec![],
            band_configs: vec![BandCompressorConfig {
                threshold_db: -12.0,
                ratio: 10.0,
                attack_secs: 0.001,
                release_secs: 0.05,
                makeup_gain_db: 0.0,
                ..Default::default()
            }],
            output_gain_db: 0.0,
            sample_rate: 48_000.0,
        };
        let mut mbc = MultibandCompressor::new(config);

        // Let the compressor settle
        for _ in 0..5000 {
            mbc.process_sample(0.9);
        }

        // After settling, check that gain reduction is being applied
        let gr = mbc.band_gain_reduction_db(0);
        assert!(
            gr > 0.0,
            "compressor should show gain reduction on loud signal, got {gr}"
        );
    }
}
