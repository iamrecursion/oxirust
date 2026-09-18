//! Channel vocoder implementation.
//!
//! Supports 4–64 analysis/synthesis bands for high-resolution vocoding.
//! Bands are logarithmically spaced from 80 Hz to min(18 kHz, Nyquist*0.9).
//! The bandpass resonance (Q) scales with band count so that adjacent bands
//! maintain good spectral separation even at 64 bands.
//!
//! ## Extended API
//!
//! [`VocoderChannelBank`] exposes a `with_config` constructor that accepts the
//! new [`VocoderConfig2`] (with [`FrequencyScale`]) for callers that need explicit
//! band-frequency control.  The legacy [`Vocoder`] struct is unchanged.

use crate::{
    filter::{FilterMode, StateVariableConfig, StateVariableFilter},
    utils::EnvelopeFollower,
    AudioEffect,
};

/// Frequency scale used to distribute vocoder bands across the spectrum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrequencyScale {
    /// Uniform linear spacing: `f_k = f_low + k * (f_high - f_low) / (N-1)`.
    Linear,
    /// Mel-like logarithmic spacing (better formant separation for voice):
    /// `f_k = f_low * (f_high / f_low)^(k / (N-1))`.
    Logarithmic,
    /// Bark critical-band scale (24 bands covering human auditory range).
    /// Band centre frequencies approximate ISO 532 Bark values.
    BarkScale,
}

/// Extended vocoder configuration used by [`VocoderChannelBank`].
#[derive(Debug, Clone)]
pub struct VocoderConfig2 {
    /// Number of frequency bands (clamped to `[8, 64]`).
    pub num_bands: usize,
    /// Lowest band centre frequency in Hz (default: `80.0`).
    pub low_freq_hz: f32,
    /// Highest band centre frequency in Hz (default: `8000.0`).
    pub high_freq_hz: f32,
    /// Frequency spacing scheme.
    pub freq_scale: FrequencyScale,
    /// Envelope follower attack time in milliseconds (default `5.0`).
    pub attack_ms: f32,
    /// Envelope follower release time in milliseconds (default `50.0`).
    pub release_ms: f32,
}

impl Default for VocoderConfig2 {
    fn default() -> Self {
        Self {
            num_bands: 16,
            low_freq_hz: 80.0,
            high_freq_hz: 8_000.0,
            freq_scale: FrequencyScale::Logarithmic,
            attack_ms: 5.0,
            release_ms: 50.0,
        }
    }
}

/// Known Bark-scale centre frequencies (Hz) for 24 critical bands.
///
/// Values from Zwicker (1961) / ISO 532 Bark approximation.
const BARK_CENTERS_HZ: [f32; 24] = [
    50.0, 150.0, 250.0, 350.0, 450.0, 570.0, 700.0, 840.0, 1_000.0, 1_170.0, 1_370.0, 1_600.0,
    1_850.0, 2_150.0, 2_500.0, 2_900.0, 3_400.0, 4_000.0, 4_800.0, 5_800.0, 7_000.0, 8_500.0,
    10_500.0, 13_500.0,
];

/// Compute band centre frequencies for a given config.
///
/// Returns a `Vec` of length `num_bands` with frequencies in Hz.
fn compute_band_freqs(config: &VocoderConfig2, sample_rate: u32) -> Vec<f32> {
    let n = config.num_bands.clamp(8, 64);
    let nyquist_safe = sample_rate as f32 * 0.45;
    let f_low = config.low_freq_hz.clamp(20.0, 2_000.0);
    let f_high = config.high_freq_hz.min(nyquist_safe).max(f_low * 2.0);

    match config.freq_scale {
        FrequencyScale::Linear => {
            let step = (f_high - f_low) / (n - 1).max(1) as f32;
            (0..n).map(|k| f_low + k as f32 * step).collect()
        }
        FrequencyScale::Logarithmic => {
            // f_k = f_low * (f_high / f_low)^(k / (N-1))
            let ratio = (f_high / f_low).ln();
            (0..n)
                .map(|k| {
                    let t = k as f32 / (n - 1).max(1) as f32;
                    f_low * (ratio * t).exp()
                })
                .collect()
        }
        FrequencyScale::BarkScale => {
            // Use the first `n` Bark centres, clamped to the configured range.
            // If n > 24 we fill remaining bands logarithmically.
            let bark_count = n.min(BARK_CENTERS_HZ.len());
            let mut freqs: Vec<f32> = BARK_CENTERS_HZ[..bark_count]
                .iter()
                .map(|&f| f.clamp(f_low, f_high))
                .collect();
            // Fill any extra bands logarithmically beyond the Bark table.
            if n > bark_count {
                let last_bark = BARK_CENTERS_HZ[bark_count - 1];
                let extra = n - bark_count;
                let ratio = (f_high / last_bark).ln();
                for k in 1..=extra {
                    let t = k as f32 / extra as f32;
                    freqs.push((last_bark * (ratio * t).exp()).min(f_high));
                }
            }
            freqs
        }
    }
}

/// Channel-bank vocoder built from a [`VocoderConfig2`].
///
/// Unlike the legacy [`Vocoder`], this type exposes band frequencies explicitly
/// and supports all three [`FrequencyScale`] modes.
///
/// ## Example
///
/// ```ignore
/// use oximedia_effects::vocoder::channel::{VocoderChannelBank, VocoderConfig2, FrequencyScale};
///
/// let config = VocoderConfig2 {
///     num_bands: 32,
///     low_freq_hz: 80.0,
///     high_freq_hz: 8_000.0,
///     freq_scale: FrequencyScale::Logarithmic,
///     ..Default::default()
/// };
/// let mut bank = VocoderChannelBank::with_config(config, 48_000);
/// let out = bank.process(voice_sample, synth_sample);
/// ```
pub struct VocoderChannelBank {
    bands: Vec<VocoderBand>,
    /// Centre frequency (Hz) of each band — retained for inspection.
    band_freqs: Vec<f32>,
}

impl VocoderChannelBank {
    /// Construct a channel bank from a [`VocoderConfig2`].
    ///
    /// Band count is clamped to `[8, 64]`.  Filter Q scales with band count
    /// for consistent spectral separation.
    #[must_use]
    pub fn with_config(config: VocoderConfig2, sample_rate: u32) -> Self {
        let num_bands = config.num_bands.clamp(8, 64);
        let freqs = compute_band_freqs(&config, sample_rate);

        // Scale Q with band count so adjacent bands stay well-separated.
        let resonance = (1.5_f32 + num_bands as f32 / 16.0).min(12.0);
        let sr_f32 = sample_rate as f32;

        let bands: Vec<VocoderBand> = freqs
            .iter()
            .map(|&frequency| {
                let filter_config = StateVariableConfig {
                    frequency,
                    resonance,
                    mode: FilterMode::BandPass,
                };
                VocoderBand {
                    modulator_filter: StateVariableFilter::new(filter_config.clone(), sr_f32),
                    carrier_filter: StateVariableFilter::new(filter_config, sr_f32),
                    envelope: EnvelopeFollower::new(config.attack_ms, config.release_ms, sr_f32),
                }
            })
            .collect();

        Self {
            bands,
            band_freqs: freqs,
        }
    }

    /// Return the actual number of active bands.
    #[must_use]
    pub fn num_bands(&self) -> usize {
        self.bands.len()
    }

    /// Return the centre frequency of band `index` in Hz.
    ///
    /// Returns `None` if `index` is out of range.
    #[must_use]
    pub fn band_freq_hz(&self, index: usize) -> Option<f32> {
        self.band_freqs.get(index).copied()
    }

    /// Process one sample pair: `modulator` (e.g. voice) and `carrier` (e.g. synth).
    pub fn process(&mut self, modulator: f32, carrier: f32) -> f32 {
        let mut output = 0.0_f32;
        for band in &mut self.bands {
            let mod_filtered = band.modulator_filter.process_sample(modulator);
            let envelope = band.envelope.process(mod_filtered);
            let car_filtered = band.carrier_filter.process_sample(carrier);
            output += car_filtered * envelope;
        }
        let scale = 1.0 / self.bands.len() as f32;
        output * scale
    }

    /// Reset all filter and envelope states.
    pub fn clear_state(&mut self) {
        for band in &mut self.bands {
            band.modulator_filter.reset();
            band.carrier_filter.reset();
            band.envelope.reset();
        }
    }
}

impl AudioEffect for VocoderChannelBank {
    const EFFECT_ID: &'static str = "vocoder_channel_bank";

    fn process_sample(&mut self, input: f32) -> f32 {
        self.process(input, input)
    }

    fn reset(&mut self) {
        self.clear_state();
    }
}

// ── Legacy API (unchanged) ──────────────────────────────────────────────────

/// Vocoder configuration.
#[derive(Debug, Clone)]
pub struct VocoderConfig {
    /// Number of frequency bands (clamped to `[4, 64]`).
    ///
    /// Common values:
    /// - 16 — traditional vocoder
    /// - 32 — high-resolution vocoding (good intelligibility)
    /// - 64 — studio-quality formant accuracy
    pub bands: usize,
    /// Envelope follower attack time in milliseconds.
    pub attack_ms: f32,
    /// Envelope follower release time in milliseconds.
    pub release_ms: f32,
    /// Minimum analysis frequency in Hz (default: `80.0`).
    pub min_freq: f32,
    /// Maximum analysis frequency in Hz (default: `18000.0`, capped at Nyquist·0.9).
    pub max_freq: f32,
}

impl Default for VocoderConfig {
    fn default() -> Self {
        Self {
            bands: 32,
            attack_ms: 5.0,
            release_ms: 50.0,
            min_freq: 80.0,
            max_freq: 18_000.0,
        }
    }
}

/// A single analysis/synthesis band.
struct VocoderBand {
    /// Bandpass filter applied to the modulator signal.
    modulator_filter: StateVariableFilter,
    /// Bandpass filter applied to the carrier signal (same frequency).
    carrier_filter: StateVariableFilter,
    /// Envelope follower that tracks modulator band energy.
    envelope: EnvelopeFollower,
}

/// Channel vocoder effect.
///
/// Imposes the spectral envelope of one signal (the *modulator*, typically
/// a voice) onto another (the *carrier*, e.g. a synthesizer pad).
///
/// ## Band count guidance
///
/// | Bands | Use-case |
/// |------:|----------|
/// |  4–8  | Robotic / heavy processing |
/// | 12–16 | Classic vocoder sound |
/// | 24–32 | High intelligibility speech vocoding |
/// | 48–64 | Studio-quality formant preservation |
///
/// ## Example
///
/// ```ignore
/// use oximedia_effects::vocoder::{Vocoder, VocoderConfig};
///
/// let config = VocoderConfig { bands: 32, ..Default::default() };
/// let mut vocoder = Vocoder::new(config, 48000.0);
/// let out = vocoder.process(voice_sample, synth_sample);
/// ```
pub struct Vocoder {
    bands: Vec<VocoderBand>,
    #[allow(dead_code)]
    config: VocoderConfig,
}

impl Vocoder {
    /// Create a new vocoder.
    ///
    /// Band count is clamped to `[4, 64]`. Bandpass Q is automatically scaled
    /// to maintain spectral separation:
    ///
    /// ```text
    /// Q = clamp(1.5 + num_bands / 16, 1.5, 12.0)
    /// ```
    #[must_use]
    pub fn new(config: VocoderConfig, sample_rate: f32) -> Self {
        let num_bands = config.bands.clamp(4, 64);

        let min_freq = config.min_freq.clamp(20.0, 2000.0);
        let nyquist_safe = sample_rate * 0.45;
        let max_freq = config.max_freq.min(nyquist_safe).max(min_freq * 2.0);

        // Scale resonance (Q) with band count for good separation.
        #[allow(clippy::cast_precision_loss)]
        let resonance = (1.5_f32 + num_bands as f32 / 16.0).min(12.0);

        let bands: Vec<VocoderBand> = (0..num_bands)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let ratio = if num_bands > 1 {
                    i as f32 / (num_bands - 1) as f32
                } else {
                    0.5
                };

                // Logarithmic frequency spacing.
                let frequency = min_freq * (max_freq / min_freq).powf(ratio);

                let filter_config = StateVariableConfig {
                    frequency,
                    resonance,
                    mode: FilterMode::BandPass,
                };

                VocoderBand {
                    modulator_filter: StateVariableFilter::new(filter_config.clone(), sample_rate),
                    carrier_filter: StateVariableFilter::new(filter_config, sample_rate),
                    envelope: EnvelopeFollower::new(
                        config.attack_ms,
                        config.release_ms,
                        sample_rate,
                    ),
                }
            })
            .collect();

        Self { bands, config }
    }

    /// Return the actual number of analysis/synthesis bands in use.
    #[must_use]
    pub fn num_bands(&self) -> usize {
        self.bands.len()
    }

    /// Process one sample pair: `modulator` (e.g. voice) and `carrier` (e.g. synth).
    ///
    /// The modulator's spectral envelope is extracted per band and applied to
    /// the carrier. Output is normalized by band count so amplitude is stable
    /// regardless of how many bands are active.
    pub fn process(&mut self, modulator: f32, carrier: f32) -> f32 {
        let mut output = 0.0_f32;

        for band in &mut self.bands {
            // 1. Filter modulator through analysis bandpass.
            let mod_filtered = band.modulator_filter.process_sample(modulator);

            // 2. Track modulator band amplitude with the envelope follower.
            let envelope = band.envelope.process(mod_filtered);

            // 3. Filter carrier through synthesis bandpass (same frequency).
            let car_filtered = band.carrier_filter.process_sample(carrier);

            // 4. Scale carrier by modulator envelope → spectrally shaped output.
            output += car_filtered * envelope;
        }

        // Normalize by band count to keep output level independent of num_bands.
        #[allow(clippy::cast_precision_loss)]
        let scale = 1.0 / self.bands.len() as f32;
        output * scale
    }
}

impl AudioEffect for Vocoder {
    const EFFECT_ID: &'static str = "vocoder";

    /// Process a single mono sample using input as both modulator and carrier.
    fn process_sample(&mut self, input: f32) -> f32 {
        self.process(input, input)
    }

    fn reset(&mut self) {
        for band in &mut self.bands {
            band.modulator_filter.reset();
            band.carrier_filter.reset();
            band.envelope.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocoder_default_process() {
        let config = VocoderConfig::default();
        let mut vocoder = Vocoder::new(config, 48000.0);
        let output = vocoder.process(0.5, 0.3);
        assert!(output.is_finite(), "output must be finite: {output}");
    }

    #[test]
    fn test_vocoder_default_band_count() {
        let config = VocoderConfig::default();
        let vocoder = Vocoder::new(config, 48000.0);
        assert_eq!(vocoder.num_bands(), 32, "default should be 32 bands");
    }

    #[test]
    fn test_vocoder_32_bands() {
        let config = VocoderConfig {
            bands: 32,
            ..Default::default()
        };
        let vocoder = Vocoder::new(config, 48000.0);
        assert_eq!(vocoder.num_bands(), 32);
    }

    #[test]
    fn test_vocoder_64_bands() {
        let config = VocoderConfig {
            bands: 64,
            ..Default::default()
        };
        let mut vocoder = Vocoder::new(config, 48000.0);
        assert_eq!(vocoder.num_bands(), 64);
        let out = vocoder.process(0.3, 0.7);
        assert!(out.is_finite(), "64-band output must be finite: {out}");
    }

    #[test]
    fn test_vocoder_clamp_max() {
        let config = VocoderConfig {
            bands: 128,
            ..Default::default()
        };
        let vocoder = Vocoder::new(config, 48000.0);
        assert_eq!(vocoder.num_bands(), 64, "bands must clamp at 64");
    }

    #[test]
    fn test_vocoder_clamp_min() {
        let config = VocoderConfig {
            bands: 1,
            ..Default::default()
        };
        let vocoder = Vocoder::new(config, 48000.0);
        assert_eq!(vocoder.num_bands(), 4, "bands must clamp to minimum 4");
    }

    #[test]
    fn test_vocoder_reset() {
        let config = VocoderConfig {
            bands: 16,
            ..Default::default()
        };
        let mut vocoder = Vocoder::new(config, 48000.0);
        for _ in 0..1000 {
            vocoder.process(0.9, 0.9);
        }
        vocoder.reset();
        let out = vocoder.process(0.0, 0.0);
        assert!(
            out.abs() < 1e-6,
            "output after reset on silence must be ~0: {out}"
        );
    }

    #[test]
    fn test_vocoder_mono_process_sample() {
        let config = VocoderConfig::default();
        let mut vocoder = Vocoder::new(config, 48000.0);
        let out = vocoder.process_sample(0.5);
        assert!(out.is_finite());
    }

    #[test]
    fn test_vocoder_output_finite_bulk() {
        let config = VocoderConfig {
            bands: 32,
            ..Default::default()
        };
        let mut vocoder = Vocoder::new(config, 48000.0);
        use std::f32::consts::TAU;
        for i in 0..4800 {
            let mod_s = (i as f32 * TAU * 300.0 / 48000.0).sin() * 0.5;
            let car_s = (i as f32 * TAU * 440.0 / 48000.0).sin() * 0.8;
            let out = vocoder.process(mod_s, car_s);
            assert!(out.is_finite(), "output at sample {i} not finite: {out}");
        }
    }

    // ── VocoderChannelBank / VocoderConfig2 / FrequencyScale tests ───────────

    #[test]
    fn test_vocoder_32_bands_no_panic() {
        // 32-band vocoder: process 1024 samples, assert no panic and finite output.
        let config = VocoderConfig2 {
            num_bands: 32,
            ..Default::default()
        };
        let mut bank = VocoderChannelBank::with_config(config, 48_000);
        assert_eq!(bank.num_bands(), 32);
        use std::f32::consts::TAU;
        for i in 0..1024_usize {
            let mod_s = (i as f32 * TAU * 250.0 / 48_000.0).sin() * 0.5;
            let car_s = (i as f32 * TAU * 440.0 / 48_000.0).sin() * 0.8;
            let out = bank.process(mod_s, car_s);
            assert!(
                out.is_finite(),
                "32-band output not finite at sample {i}: {out}"
            );
        }
    }

    #[test]
    fn test_vocoder_64_bands_no_panic() {
        // 64-band vocoder: process 1024 samples, assert no panic and finite output.
        let config = VocoderConfig2 {
            num_bands: 64,
            low_freq_hz: 80.0,
            high_freq_hz: 8_000.0,
            freq_scale: FrequencyScale::Logarithmic,
            ..Default::default()
        };
        let mut bank = VocoderChannelBank::with_config(config, 48_000);
        assert_eq!(bank.num_bands(), 64);
        use std::f32::consts::TAU;
        for i in 0..1024_usize {
            let mod_s = (i as f32 * TAU * 300.0 / 48_000.0).sin() * 0.5;
            let car_s = (i as f32 * TAU * 880.0 / 48_000.0).sin() * 0.6;
            let out = bank.process(mod_s, car_s);
            assert!(
                out.is_finite(),
                "64-band output not finite at sample {i}: {out}"
            );
        }
    }

    #[test]
    fn test_vocoder_log_spacing_correct() {
        // 8 log-spaced bands from 80 Hz to 8000 Hz.
        // band[0] ≈ 80 Hz, band[7] ≈ 8000 Hz (within 1%).
        let config = VocoderConfig2 {
            num_bands: 8,
            low_freq_hz: 80.0,
            high_freq_hz: 8_000.0,
            freq_scale: FrequencyScale::Logarithmic,
            ..Default::default()
        };
        let bank = VocoderChannelBank::with_config(config, 48_000);
        assert_eq!(bank.num_bands(), 8);

        let f0 = bank.band_freq_hz(0).expect("band 0 must exist");
        let f7 = bank.band_freq_hz(7).expect("band 7 must exist");

        let tol = 0.01; // 1 %
        assert!(
            (f0 - 80.0).abs() / 80.0 <= tol,
            "band[0] should be ~80 Hz, got {f0:.2} Hz"
        );
        assert!(
            (f7 - 8_000.0).abs() / 8_000.0 <= tol,
            "band[7] should be ~8000 Hz, got {f7:.2} Hz"
        );
    }

    #[test]
    fn test_vocoder_bark_scale() {
        // 24 Bark-scale bands: verify that the first few band frequencies
        // approximately match the Bark centre-frequency table.
        let config = VocoderConfig2 {
            num_bands: 24,
            low_freq_hz: 20.0,
            high_freq_hz: 20_000.0,
            freq_scale: FrequencyScale::BarkScale,
            ..Default::default()
        };
        let bank = VocoderChannelBank::with_config(config, 96_000);
        assert_eq!(bank.num_bands(), 24);

        // Check a selection of well-known Bark centres (Hz).
        let expected = [(0, 50.0_f32), (4, 450.0), (8, 1_000.0), (11, 1_600.0)];
        for (idx, expected_hz) in expected {
            let actual = bank.band_freq_hz(idx).expect("band must exist");
            let rel_err = (actual - expected_hz).abs() / expected_hz;
            assert!(
                rel_err < 0.05,
                "Bark band[{idx}] expected ~{expected_hz:.0} Hz, got {actual:.2} Hz (rel err {rel_err:.3})"
            );
        }
    }
}
