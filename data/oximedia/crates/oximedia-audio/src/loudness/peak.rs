//! True peak detection for preventing inter-sample clipping.
//!
//! Implements true peak measurement using oversampling as specified
//! in ITU-R BS.1770-4 and EBU R128.

#![forbid(unsafe_code)]
#![allow(clippy::cast_lossless)]

use std::f64::consts::PI;

/// Oversampling factor for true peak detection.
///
/// ITU-R BS.1770-4 requires at least 4x oversampling.
const OVERSAMPLE_FACTOR: usize = 4;

/// True peak detector with oversampling.
///
/// Detects peaks that occur between samples (inter-sample peaks)
/// by upsampling the signal using a polyphase FIR filter.
#[derive(Clone, Debug)]
pub struct TruePeakDetector {
    /// Sample rate in Hz.
    sample_rate: f64,
    /// Number of audio channels.
    channels: usize,
    /// Per-channel peak detectors.
    detectors: Vec<ChannelPeakDetector>,
}

impl TruePeakDetector {
    /// Create a new true peak detector.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz
    /// * `channels` - Number of audio channels
    #[must_use]
    pub fn new(sample_rate: f64, channels: usize) -> Self {
        let detectors = (0..channels)
            .map(|_| ChannelPeakDetector::new(OVERSAMPLE_FACTOR))
            .collect();

        Self {
            sample_rate,
            channels,
            detectors,
        }
    }

    /// Process interleaved samples and detect true peaks.
    ///
    /// # Arguments
    ///
    /// * `samples` - Interleaved audio samples
    ///
    /// # Returns
    ///
    /// Maximum true peak value across all channels (linear scale, 0.0 to 1.0+)
    pub fn process_interleaved(&mut self, samples: &[f64]) -> f64 {
        if samples.is_empty() || self.channels == 0 {
            return 0.0;
        }

        let frames = samples.len() / self.channels;
        let mut max_peak: f64 = 0.0;

        for frame in 0..frames {
            for ch in 0..self.channels {
                let idx = frame * self.channels + ch;
                let sample = samples.get(idx).copied().unwrap_or(0.0);
                let peak = self.detectors[ch].process(sample);
                max_peak = max_peak.max(peak);
            }
        }

        max_peak
    }

    /// Process planar samples and detect true peaks.
    ///
    /// # Arguments
    ///
    /// * `channels` - Slice of per-channel sample buffers
    ///
    /// # Returns
    ///
    /// Maximum true peak value across all channels (linear scale, 0.0 to 1.0+)
    pub fn process_planar(&mut self, channels: &[&[f64]]) -> f64 {
        let mut max_peak: f64 = 0.0;

        for (ch_idx, samples) in channels.iter().enumerate() {
            if ch_idx < self.detectors.len() {
                for &sample in samples.iter() {
                    let peak = self.detectors[ch_idx].process(sample);
                    max_peak = max_peak.max(peak);
                }
            }
        }

        max_peak
    }

    /// Get the current peak for a specific channel.
    ///
    /// # Arguments
    ///
    /// * `channel` - Channel index
    ///
    /// # Returns
    ///
    /// Current peak value (linear scale, 0.0 to 1.0+)
    #[must_use]
    pub fn get_channel_peak(&self, channel: usize) -> f64 {
        self.detectors.get(channel).map_or(0.0, |d| d.peak)
    }

    /// Get peaks for all channels.
    ///
    /// # Returns
    ///
    /// Vector of peak values (linear scale, 0.0 to 1.0+)
    #[must_use]
    pub fn get_all_peaks(&self) -> Vec<f64> {
        self.detectors.iter().map(|d| d.peak).collect()
    }

    /// Reset all peak detectors.
    pub fn reset(&mut self) {
        for detector in &mut self.detectors {
            detector.reset();
        }
    }

    /// Convert linear peak to dBTP (dB True Peak).
    ///
    /// # Arguments
    ///
    /// * `linear` - Linear peak value
    ///
    /// # Returns
    ///
    /// Peak level in dBTP
    #[must_use]
    pub fn linear_to_dbtp(linear: f64) -> f64 {
        if linear <= 0.0 {
            f64::NEG_INFINITY
        } else {
            20.0 * linear.log10()
        }
    }

    /// Convert dBTP to linear peak.
    ///
    /// # Arguments
    ///
    /// * `dbtp` - Peak level in dBTP
    ///
    /// # Returns
    ///
    /// Linear peak value
    #[must_use]
    pub fn dbtp_to_linear(dbtp: f64) -> f64 {
        if dbtp.is_infinite() && dbtp.is_sign_negative() {
            0.0
        } else {
            10.0_f64.powf(dbtp / 20.0)
        }
    }

    /// Get the oversampling factor.
    #[must_use]
    pub fn oversample_factor() -> usize {
        OVERSAMPLE_FACTOR
    }
}

/// Per-channel true peak detector.
#[derive(Clone, Debug)]
struct ChannelPeakDetector {
    /// Oversampling factor.
    oversample: usize,
    /// Polyphase filter banks for oversampling.
    filters: Vec<OversampleFilter>,
    /// Current peak value.
    peak: f64,
}

impl ChannelPeakDetector {
    /// Create a new channel peak detector.
    fn new(oversample: usize) -> Self {
        let filters = (0..oversample)
            .map(|phase| OversampleFilter::new(phase, oversample))
            .collect();

        Self {
            oversample,
            filters,
            peak: 0.0,
        }
    }

    /// Process a single sample and update peak.
    fn process(&mut self, sample: f64) -> f64 {
        // Feed sample to all polyphase filters
        for filter in &mut self.filters {
            let upsampled = filter.process(sample);
            let abs_val = upsampled.abs();
            self.peak = self.peak.max(abs_val);
        }

        self.peak
    }

    /// Reset peak detector.
    fn reset(&mut self) {
        self.peak = 0.0;
        for filter in &mut self.filters {
            filter.reset();
        }
    }
}

/// Polyphase FIR filter for oversampling.
///
/// Uses a windowed-sinc interpolation filter to upsample the signal.
#[derive(Clone, Debug)]
struct OversampleFilter {
    /// Filter coefficients.
    coeffs: Vec<f64>,
    /// Delay line (sample history).
    delay_line: Vec<f64>,
    /// Write position in delay line.
    write_pos: usize,
}

impl OversampleFilter {
    /// Create a new oversampling filter for a specific phase.
    ///
    /// # Arguments
    ///
    /// * `phase` - Phase index (0 to oversample-1)
    /// * `oversample` - Oversampling factor
    fn new(phase: usize, oversample: usize) -> Self {
        // Design windowed-sinc filter
        // Filter length: 12 taps per polyphase branch (48 total for 4x)
        let taps_per_phase = 12;
        let coeffs = Self::design_filter(phase, oversample, taps_per_phase);
        let delay_line = vec![0.0; taps_per_phase];

        Self {
            coeffs,
            delay_line,
            write_pos: 0,
        }
    }

    /// Design windowed-sinc interpolation filter.
    fn design_filter(phase: usize, oversample: usize, length: usize) -> Vec<f64> {
        let mut coeffs = Vec::with_capacity(length);
        let center = (length - 1) as f64 / 2.0;

        for i in 0..length {
            let x = i as f64 - center + phase as f64 / oversample as f64;

            // Sinc function
            let sinc = if x.abs() < 1e-10 {
                1.0
            } else {
                let pi_x = PI * x;
                pi_x.sin() / pi_x
            };

            // Hamming window
            let window = 0.54 - 0.46 * (2.0 * PI * i as f64 / (length - 1) as f64).cos();

            coeffs.push(sinc * window);
        }

        // Normalize
        let sum: f64 = coeffs.iter().sum();
        if sum.abs() > 1e-10 {
            for coeff in &mut coeffs {
                *coeff /= sum;
            }
        }

        coeffs
    }

    /// Process a single sample through the filter.
    fn process(&mut self, sample: f64) -> f64 {
        // Add sample to delay line
        self.delay_line[self.write_pos] = sample;
        self.write_pos = (self.write_pos + 1) % self.delay_line.len();

        // Convolve with filter coefficients
        let mut output = 0.0;
        let mut read_pos = self.write_pos;

        for &coeff in &self.coeffs {
            output += coeff * self.delay_line[read_pos];
            read_pos = (read_pos + 1) % self.delay_line.len();
        }

        output
    }

    /// Reset filter state.
    fn reset(&mut self) {
        self.delay_line.fill(0.0);
        self.write_pos = 0;
    }
}

/// Simple peak detector (sample peak, not true peak).
///
/// Detects the maximum absolute sample value without oversampling.
/// This is faster but may miss inter-sample peaks.
#[derive(Clone, Debug, Default)]
pub struct SamplePeakDetector {
    /// Per-channel peak values.
    peaks: Vec<f64>,
}

impl SamplePeakDetector {
    /// Create a new sample peak detector.
    ///
    /// # Arguments
    ///
    /// * `channels` - Number of audio channels
    #[must_use]
    pub fn new(channels: usize) -> Self {
        Self {
            peaks: vec![0.0; channels],
        }
    }

    /// Process interleaved samples.
    ///
    /// # Arguments
    ///
    /// * `samples` - Interleaved audio samples
    /// * `channels` - Number of channels
    pub fn process_interleaved(&mut self, samples: &[f64], channels: usize) {
        if channels == 0 || channels != self.peaks.len() {
            return;
        }

        let frames = samples.len() / channels;

        for frame in 0..frames {
            for ch in 0..channels {
                let idx = frame * channels + ch;
                if let Some(&sample) = samples.get(idx) {
                    self.peaks[ch] = self.peaks[ch].max(sample.abs());
                }
            }
        }
    }

    /// Process planar samples.
    ///
    /// # Arguments
    ///
    /// * `channels` - Slice of per-channel sample buffers
    pub fn process_planar(&mut self, channels: &[&[f64]]) {
        for (ch_idx, samples) in channels.iter().enumerate() {
            if ch_idx < self.peaks.len() {
                for &sample in samples.iter() {
                    self.peaks[ch_idx] = self.peaks[ch_idx].max(sample.abs());
                }
            }
        }
    }

    /// Get peak for a specific channel.
    ///
    /// # Arguments
    ///
    /// * `channel` - Channel index
    ///
    /// # Returns
    ///
    /// Peak value (linear scale, 0.0 to 1.0+)
    #[must_use]
    pub fn get_peak(&self, channel: usize) -> f64 {
        self.peaks.get(channel).copied().unwrap_or(0.0)
    }

    /// Get maximum peak across all channels.
    ///
    /// # Returns
    ///
    /// Maximum peak value (linear scale, 0.0 to 1.0+)
    #[must_use]
    pub fn max_peak(&self) -> f64 {
        self.peaks.iter().copied().fold(0.0, f64::max)
    }

    /// Get all channel peaks.
    ///
    /// # Returns
    ///
    /// Vector of peak values (linear scale, 0.0 to 1.0+)
    #[must_use]
    pub fn get_all_peaks(&self) -> &[f64] {
        &self.peaks
    }

    /// Reset all peaks to zero.
    pub fn reset(&mut self) {
        self.peaks.fill(0.0);
    }
}

// ---------------------------------------------------------------------------
// True Peak Limiter
// ---------------------------------------------------------------------------

/// True peak limiter configuration.
///
/// Uses 4x oversampled peak detection to catch inter-sample peaks
/// and applies transparent gain reduction to keep the output below
/// the ceiling.
#[derive(Clone, Debug)]
pub struct TruePeakLimiterConfig {
    /// Ceiling in dBTP (typically -1.0 for broadcast).
    pub ceiling_dbtp: f64,
    /// Release time in seconds.
    pub release_secs: f64,
    /// Sample rate in Hz.
    pub sample_rate: f64,
    /// Number of channels.
    pub channels: usize,
    /// Look-ahead in samples (delay introduced for transparent limiting).
    pub lookahead_samples: usize,
}

impl Default for TruePeakLimiterConfig {
    fn default() -> Self {
        Self {
            ceiling_dbtp: -1.0,
            release_secs: 0.1,
            sample_rate: 48_000.0,
            channels: 2,
            lookahead_samples: 8,
        }
    }
}

/// Per-channel state for the true peak limiter.
#[derive(Clone, Debug)]
struct LimiterChannelState {
    /// 4x oversampled peak detector.
    peak_detector: ChannelPeakDetector,
    /// Delay buffer for look-ahead.
    delay_buffer: Vec<f64>,
    /// Write position in the delay buffer.
    write_pos: usize,
    /// Current gain envelope (linear, 0.0..=1.0).
    gain_envelope: f64,
}

impl LimiterChannelState {
    fn new(oversample: usize, lookahead: usize) -> Self {
        Self {
            peak_detector: ChannelPeakDetector::new(oversample),
            delay_buffer: vec![0.0; lookahead.max(1)],
            write_pos: 0,
            gain_envelope: 1.0,
        }
    }

    fn reset(&mut self) {
        self.peak_detector.reset();
        self.delay_buffer.fill(0.0);
        self.write_pos = 0;
        self.gain_envelope = 1.0;
    }
}

/// True peak limiter with 4x oversampled detection.
///
/// This limiter uses polyphase FIR upsampling (identical to [`TruePeakDetector`])
/// to detect inter-sample peaks that would clip after D/A conversion. When a
/// peak exceeds the ceiling, transparent gain reduction is applied via a
/// look-ahead delay buffer.
///
/// # Algorithm
///
/// 1. Each input sample is fed through a 4x polyphase oversampler.
/// 2. The maximum oversampled value across all phases determines the true peak.
/// 3. If the true peak exceeds the ceiling, a gain factor is computed.
/// 4. The gain envelope is smoothed with a one-pole release filter.
/// 5. The *delayed* (look-ahead) sample is multiplied by the gain.
///
/// This design avoids distortion artefacts caused by instantaneous limiting
/// and is compliant with EBU R128 / ITU-R BS.1770-4 true peak requirements.
#[derive(Clone, Debug)]
pub struct TruePeakLimiter {
    /// Limiter configuration.
    config: TruePeakLimiterConfig,
    /// Per-channel state.
    channel_states: Vec<LimiterChannelState>,
    /// Ceiling in linear amplitude.
    ceiling_linear: f64,
    /// Release coefficient (one-pole IIR).
    release_coeff: f64,
    /// Current gain reduction in dB (positive = reduction applied).
    last_gain_reduction_db: f64,
}

impl TruePeakLimiter {
    /// Create a new true peak limiter.
    ///
    /// # Arguments
    ///
    /// * `config` - Limiter configuration
    #[must_use]
    pub fn new(config: TruePeakLimiterConfig) -> Self {
        let ceiling_linear = TruePeakDetector::dbtp_to_linear(config.ceiling_dbtp);
        let release_coeff = if config.release_secs > 0.0 && config.sample_rate > 0.0 {
            (-1.0 / (config.release_secs * config.sample_rate)).exp()
        } else {
            0.0
        };

        let channel_states = (0..config.channels)
            .map(|_| LimiterChannelState::new(OVERSAMPLE_FACTOR, config.lookahead_samples))
            .collect();

        Self {
            config,
            channel_states,
            ceiling_linear,
            release_coeff,
            last_gain_reduction_db: 0.0,
        }
    }

    /// Process a single mono sample (channel 0).
    ///
    /// For multi-channel processing, use [`process_interleaved`](Self::process_interleaved).
    pub fn process_sample(&mut self, input: f64) -> f64 {
        if self.channel_states.is_empty() {
            return input;
        }
        self.process_channel_sample(0, input)
    }

    /// Internal: process one sample for a specific channel.
    fn process_channel_sample(&mut self, ch: usize, input: f64) -> f64 {
        let state = match self.channel_states.get_mut(ch) {
            Some(s) => s,
            None => return input,
        };

        // Detect true peak via 4x oversampled polyphase filters
        let mut true_peak: f64 = 0.0;
        for filter in &mut state.peak_detector.filters {
            let oversampled = filter.process(input);
            true_peak = true_peak.max(oversampled.abs());
        }

        // Compute required gain
        let target_gain = if true_peak > self.ceiling_linear {
            self.ceiling_linear / true_peak
        } else {
            1.0
        };

        // Smooth gain envelope (instant attack, smooth release)
        if target_gain < state.gain_envelope {
            state.gain_envelope = target_gain; // instant attack
        } else {
            state.gain_envelope =
                self.release_coeff * state.gain_envelope + (1.0 - self.release_coeff) * target_gain;
        }

        // Read from look-ahead delay
        let delayed = state.delay_buffer[state.write_pos];
        state.delay_buffer[state.write_pos] = input;
        state.write_pos = (state.write_pos + 1) % state.delay_buffer.len();

        let output = delayed * state.gain_envelope;

        // Track gain reduction
        if state.gain_envelope > 0.0 && state.gain_envelope < 1.0 {
            self.last_gain_reduction_db = -20.0 * state.gain_envelope.log10();
        } else {
            self.last_gain_reduction_db = 0.0;
        }

        output
    }

    /// Process interleaved multi-channel samples in-place.
    ///
    /// # Arguments
    ///
    /// * `samples` - Interleaved audio samples (modified in-place)
    pub fn process_interleaved(&mut self, samples: &mut [f64]) {
        if self.config.channels == 0 {
            return;
        }
        let channels = self.config.channels;
        let frames = samples.len() / channels;

        for frame in 0..frames {
            // First pass: find the maximum true peak across all channels
            let mut max_true_peak: f64 = 0.0;
            for ch in 0..channels {
                let idx = frame * channels + ch;
                let input = samples.get(idx).copied().unwrap_or(0.0);

                if let Some(state) = self.channel_states.get_mut(ch) {
                    for filter in &mut state.peak_detector.filters {
                        let oversampled = filter.process(input);
                        max_true_peak = max_true_peak.max(oversampled.abs());
                    }
                }
            }

            // Compute linked gain (same for all channels to preserve stereo image)
            let target_gain = if max_true_peak > self.ceiling_linear {
                self.ceiling_linear / max_true_peak
            } else {
                1.0
            };

            // Apply linked gain envelope to all channels
            for ch in 0..channels {
                let idx = frame * channels + ch;
                if let Some(state) = self.channel_states.get_mut(ch) {
                    if target_gain < state.gain_envelope {
                        state.gain_envelope = target_gain;
                    } else {
                        state.gain_envelope = self.release_coeff * state.gain_envelope
                            + (1.0 - self.release_coeff) * target_gain;
                    }

                    let delayed = state.delay_buffer[state.write_pos];
                    if let Some(sample) = samples.get(idx) {
                        state.delay_buffer[state.write_pos] = *sample;
                    }
                    state.write_pos = (state.write_pos + 1) % state.delay_buffer.len();

                    if let Some(out) = samples.get_mut(idx) {
                        *out = delayed * state.gain_envelope;
                    }
                }
            }

            // Track gain reduction from first channel
            if let Some(state) = self.channel_states.first() {
                if state.gain_envelope > 0.0 && state.gain_envelope < 1.0 {
                    self.last_gain_reduction_db = -20.0 * state.gain_envelope.log10();
                } else {
                    self.last_gain_reduction_db = 0.0;
                }
            }
        }
    }

    /// Process a mono buffer in-place.
    pub fn process_buffer(&mut self, samples: &mut [f64]) {
        for sample in samples.iter_mut() {
            *sample = self.process_sample(*sample);
        }
    }

    /// Get the last computed gain reduction in dB.
    #[must_use]
    pub fn gain_reduction_db(&self) -> f64 {
        self.last_gain_reduction_db
    }

    /// Get the ceiling in dBTP.
    #[must_use]
    pub fn ceiling_dbtp(&self) -> f64 {
        self.config.ceiling_dbtp
    }

    /// Set a new ceiling.
    pub fn set_ceiling_dbtp(&mut self, ceiling_dbtp: f64) {
        self.config.ceiling_dbtp = ceiling_dbtp;
        self.ceiling_linear = TruePeakDetector::dbtp_to_linear(ceiling_dbtp);
    }

    /// Reset the limiter state.
    pub fn reset(&mut self) {
        for state in &mut self.channel_states {
            state.reset();
        }
        self.last_gain_reduction_db = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- TruePeakDetector tests ---

    #[test]
    fn test_true_peak_detector_creation() {
        let detector = TruePeakDetector::new(48_000.0, 2);
        assert_eq!(detector.channels, 2);
        assert_eq!(detector.detectors.len(), 2);
    }

    #[test]
    fn test_true_peak_detector_silence() {
        let mut detector = TruePeakDetector::new(48_000.0, 1);
        let samples = vec![0.0_f64; 1000];
        let peak = detector.process_interleaved(&samples);
        assert!(peak.abs() < 1e-10, "silence should have zero peak");
    }

    #[test]
    fn test_true_peak_detector_dc_signal() {
        let mut detector = TruePeakDetector::new(48_000.0, 1);
        let samples = vec![0.5_f64; 5000];
        let peak = detector.process_interleaved(&samples);
        // After filter settling, peak should be near 0.5
        assert!(peak > 0.3, "DC signal peak should be near 0.5, got {peak}");
    }

    #[test]
    fn test_true_peak_detector_stereo() {
        let mut detector = TruePeakDetector::new(48_000.0, 2);
        // L=0.5, R=0.8 interleaved
        let mut samples = Vec::new();
        for _ in 0..2000 {
            samples.push(0.5);
            samples.push(0.8);
        }
        let peak = detector.process_interleaved(&samples);
        assert!(peak > 0.6, "stereo peak should detect R channel = 0.8");
    }

    #[test]
    fn test_true_peak_detector_channel_peaks() {
        let mut detector = TruePeakDetector::new(48_000.0, 2);
        let mut samples = Vec::new();
        for _ in 0..2000 {
            samples.push(0.3);
            samples.push(0.7);
        }
        let _ = detector.process_interleaved(&samples);
        let peaks = detector.get_all_peaks();
        assert_eq!(peaks.len(), 2);
        assert!(peaks[1] > peaks[0], "right channel should have higher peak");
    }

    #[test]
    fn test_true_peak_detector_planar() {
        let mut detector = TruePeakDetector::new(48_000.0, 2);
        let left = vec![0.4_f64; 2000];
        let right = vec![0.9_f64; 2000];
        let peak = detector.process_planar(&[&left, &right]);
        assert!(peak > 0.7, "planar detection should find right peak");
    }

    #[test]
    fn test_true_peak_detector_reset() {
        let mut detector = TruePeakDetector::new(48_000.0, 1);
        let samples = vec![0.8_f64; 2000];
        let _ = detector.process_interleaved(&samples);
        assert!(detector.get_channel_peak(0) > 0.0);
        detector.reset();
        assert_eq!(detector.get_channel_peak(0), 0.0);
    }

    #[test]
    fn test_dbtp_to_linear_roundtrip() {
        let linear = 0.5;
        let dbtp = TruePeakDetector::linear_to_dbtp(linear);
        let back = TruePeakDetector::dbtp_to_linear(dbtp);
        assert!((back - linear).abs() < 1e-10);
    }

    #[test]
    fn test_dbtp_zero_is_neg_inf() {
        let dbtp = TruePeakDetector::linear_to_dbtp(0.0);
        assert!(dbtp.is_infinite() && dbtp.is_sign_negative());
    }

    #[test]
    fn test_dbtp_neg_inf_to_linear_zero() {
        let linear = TruePeakDetector::dbtp_to_linear(f64::NEG_INFINITY);
        assert_eq!(linear, 0.0);
    }

    #[test]
    fn test_oversample_factor() {
        assert_eq!(TruePeakDetector::oversample_factor(), 4);
    }

    #[test]
    fn test_true_peak_inter_sample_detection() {
        // Generate a signal that has inter-sample peaks:
        // Two adjacent samples of opposite polarity close to full scale
        // will produce a sinc-interpolated peak above either sample.
        let mut detector = TruePeakDetector::new(48_000.0, 1);

        let mut samples = vec![0.0_f64; 1000];
        // Create a pattern that produces inter-sample peaks
        for i in (100..200).step_by(2) {
            samples[i] = 0.9;
            samples[i + 1] = -0.9;
        }

        let peak = detector.process_interleaved(&samples);
        // The true peak should be >= 0.9 (oversampling catches inter-sample peak)
        assert!(
            peak >= 0.89,
            "true peak should detect inter-sample peak, got {peak}"
        );
    }

    // --- SamplePeakDetector tests ---

    #[test]
    fn test_sample_peak_detector_creation() {
        let detector = SamplePeakDetector::new(2);
        assert_eq!(detector.get_all_peaks().len(), 2);
    }

    #[test]
    fn test_sample_peak_detector_interleaved() {
        let mut detector = SamplePeakDetector::new(2);
        let samples = vec![0.3, -0.5, 0.7, 0.1];
        detector.process_interleaved(&samples, 2);
        assert!((detector.get_peak(0) - 0.7).abs() < 1e-6);
        assert!((detector.get_peak(1) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_sample_peak_detector_max_peak() {
        let mut detector = SamplePeakDetector::new(2);
        let samples = vec![0.3, -0.5, 0.7, 0.1];
        detector.process_interleaved(&samples, 2);
        assert!((detector.max_peak() - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_sample_peak_detector_reset() {
        let mut detector = SamplePeakDetector::new(1);
        detector.process_interleaved(&[0.8], 1);
        detector.reset();
        assert_eq!(detector.max_peak(), 0.0);
    }

    // --- TruePeakLimiter tests ---

    #[test]
    fn test_limiter_creation() {
        let limiter = TruePeakLimiter::new(TruePeakLimiterConfig::default());
        assert!((limiter.ceiling_dbtp() - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_limiter_silence_passthrough() {
        let mut limiter = TruePeakLimiter::new(TruePeakLimiterConfig::default());
        // Process silence — should get zeros out (after look-ahead delay)
        for _ in 0..100 {
            let out = limiter.process_sample(0.0);
            assert!(out.abs() < 1e-10);
        }
    }

    #[test]
    fn test_limiter_quiet_signal_no_reduction() {
        let mut limiter = TruePeakLimiter::new(TruePeakLimiterConfig {
            ceiling_dbtp: 0.0, // 0 dBTP = linear 1.0
            lookahead_samples: 4,
            ..Default::default()
        });

        // Warm up through look-ahead delay
        for _ in 0..20 {
            limiter.process_sample(0.1);
        }

        // Quiet signal (well below ceiling) should pass with minimal change
        let out = limiter.process_sample(0.1);
        assert!(
            (out - 0.1).abs() < 0.05,
            "quiet signal should pass nearly unchanged, got {out}"
        );
    }

    #[test]
    fn test_limiter_loud_signal_is_reduced() {
        let mut limiter = TruePeakLimiter::new(TruePeakLimiterConfig {
            ceiling_dbtp: -6.0, // ~0.5 linear
            lookahead_samples: 4,
            channels: 1,
            release_secs: 0.1,
            sample_rate: 48_000.0,
        });

        // Process loud signal — skip the first few samples (look-ahead delay)
        // and measure the steady-state output after the limiter settles.
        for _ in 0..100 {
            limiter.process_sample(0.9);
        }

        // After settling, check that gain reduction is applied
        let gr = limiter.gain_reduction_db();
        assert!(
            gr > 0.0,
            "limiter should show gain reduction on loud signal, got {gr}"
        );
    }

    #[test]
    fn test_limiter_gain_reduction_on_loud_signal() {
        let mut limiter = TruePeakLimiter::new(TruePeakLimiterConfig {
            ceiling_dbtp: -6.0,
            ..Default::default()
        });

        for _ in 0..5000 {
            limiter.process_sample(0.9);
        }

        let gr = limiter.gain_reduction_db();
        assert!(
            gr > 0.0,
            "should show gain reduction on loud signal, got {gr}"
        );
    }

    #[test]
    fn test_limiter_set_ceiling() {
        let mut limiter = TruePeakLimiter::new(TruePeakLimiterConfig::default());
        limiter.set_ceiling_dbtp(-3.0);
        assert!((limiter.ceiling_dbtp() - (-3.0)).abs() < 1e-6);
    }

    #[test]
    fn test_limiter_reset() {
        let mut limiter = TruePeakLimiter::new(TruePeakLimiterConfig::default());
        for _ in 0..1000 {
            limiter.process_sample(0.8);
        }
        limiter.reset();
        assert_eq!(limiter.gain_reduction_db(), 0.0);
    }

    #[test]
    fn test_limiter_process_buffer() {
        let mut limiter = TruePeakLimiter::new(TruePeakLimiterConfig::default());
        let mut buf = vec![0.5_f64; 2000];
        limiter.process_buffer(&mut buf);
        for &s in &buf {
            assert!(s.is_finite(), "all outputs must be finite");
        }
    }

    #[test]
    fn test_limiter_interleaved_stereo() {
        let config = TruePeakLimiterConfig {
            ceiling_dbtp: -3.0,
            channels: 2,
            lookahead_samples: 4,
            ..Default::default()
        };
        let mut limiter = TruePeakLimiter::new(config);

        // L=0.9, R=0.9 interleaved
        let mut samples = vec![0.9_f64; 4000]; // 2000 frames
        limiter.process_interleaved(&mut samples);

        for &s in &samples {
            assert!(s.is_finite());
        }
    }

    #[test]
    fn test_limiter_empty_channels() {
        let config = TruePeakLimiterConfig {
            channels: 0,
            ..Default::default()
        };
        let mut limiter = TruePeakLimiter::new(config);
        let out = limiter.process_sample(0.5);
        assert!((out - 0.5).abs() < 1e-6, "no channels should passthrough");
    }

    #[test]
    fn test_limiter_output_finite_on_all_signals() {
        let mut limiter = TruePeakLimiter::new(TruePeakLimiterConfig::default());
        // Test with various signal levels
        for level in &[0.0, 0.001, 0.1, 0.5, 0.9, 1.0, 1.5, 2.0] {
            for _ in 0..500 {
                let out = limiter.process_sample(*level);
                assert!(out.is_finite(), "output must be finite for level {level}");
            }
        }
    }
}
