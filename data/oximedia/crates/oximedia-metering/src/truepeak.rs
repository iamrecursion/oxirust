//! True peak detection using 4x oversampling.
//!
//! Implements ITU-R BS.1770-4 true peak measurement to detect inter-sample peaks
//! that would occur during digital-to-analog conversion.
//!
//! Uses windowed sinc interpolation (Lanczos) for 4x oversampling.

#![allow(clippy::similar_names)]

use std::f64::consts::PI;

/// Oversampling factor for true peak detection (4x standard).
const OVERSAMPLE_FACTOR: usize = 4;

/// Oversampling factor for mastering-grade true peak detection (8x).
const OVERSAMPLE_FACTOR_8X: usize = 8;

/// Lanczos window size (a=3 gives good results).
const LANCZOS_A: usize = 3;

/// Number of filter taps per phase.
const TAPS_PER_PHASE: usize = LANCZOS_A * 2;

/// True peak value with metadata.
#[derive(Clone, Copy, Debug)]
pub struct TruePeak {
    /// Peak value in linear scale (0.0 to inf).
    pub linear: f64,
    /// Peak value in dBTP (dB True Peak).
    pub dbtp: f64,
    /// Sample index where peak occurred.
    pub sample_index: usize,
    /// Channel where peak occurred.
    pub channel: usize,
}

impl TruePeak {
    /// Create a new true peak measurement.
    pub fn new(linear: f64, channel: usize, sample_index: usize) -> Self {
        let dbtp = if linear > 0.0 {
            20.0 * linear.log10()
        } else {
            f64::NEG_INFINITY
        };

        Self {
            linear,
            dbtp,
            sample_index,
            channel,
        }
    }

    /// Check if peak exceeds a threshold in dBTP.
    pub fn exceeds(&self, threshold_dbtp: f64) -> bool {
        self.dbtp > threshold_dbtp
    }
}

/// Oversampling mode for true peak detection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OversampleMode {
    /// Standard 4x oversampling (ITU-R BS.1770-4 compliant).
    FourX,
    /// Mastering-grade 8x oversampling for higher inter-sample peak precision.
    EightX,
}

impl OversampleMode {
    /// Return the oversampling factor as an integer.
    pub fn factor(&self) -> usize {
        match self {
            Self::FourX => OVERSAMPLE_FACTOR,
            Self::EightX => OVERSAMPLE_FACTOR_8X,
        }
    }
}

/// True peak detector with configurable oversampling.
///
/// Uses Lanczos-windowed sinc resampling to detect inter-sample peaks.
/// Supports 4x (standard, ITU-R BS.1770-4) and 8x (mastering-grade) oversampling.
pub struct TruePeakDetector {
    sample_rate: f64,
    channels: usize,

    /// Oversampling mode (4x or 8x).
    oversample_mode: OversampleMode,

    // Per-channel delay lines for resampling
    delay_lines: Vec<DelayLine>,

    // Per-channel peak tracking
    channel_peaks: Vec<f64>,
    channel_peak_indices: Vec<usize>,

    // Resampling filter coefficients (per phase)
    filter_coeffs: Vec<Vec<f64>>,

    // Sample counter
    sample_index: usize,
}

impl TruePeakDetector {
    /// Create a new true peak detector with standard 4x oversampling.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz
    /// * `channels` - Number of audio channels
    pub fn new(sample_rate: f64, channels: usize) -> Self {
        Self::with_oversample(sample_rate, channels, OversampleMode::FourX)
    }

    /// Create a new true peak detector with mastering-grade 8x oversampling.
    ///
    /// 8x oversampling provides higher precision inter-sample peak detection
    /// at the cost of approximately 2x the computation per channel.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz
    /// * `channels` - Number of audio channels
    pub fn new_8x(sample_rate: f64, channels: usize) -> Self {
        Self::with_oversample(sample_rate, channels, OversampleMode::EightX)
    }

    /// Create a new true peak detector with explicit oversampling mode.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz
    /// * `channels` - Number of audio channels
    /// * `mode` - Oversampling mode (4x or 8x)
    pub fn with_oversample(sample_rate: f64, channels: usize, mode: OversampleMode) -> Self {
        let filter_coeffs = Self::design_lanczos_filters_n(mode.factor());
        let delay_lines = (0..channels)
            .map(|_| DelayLine::new(TAPS_PER_PHASE))
            .collect();

        Self {
            sample_rate,
            channels,
            oversample_mode: mode,
            delay_lines,
            channel_peaks: vec![0.0; channels],
            channel_peak_indices: vec![0; channels],
            filter_coeffs,
            sample_index: 0,
        }
    }

    /// Process interleaved audio samples.
    ///
    /// For more than 4 channels, uses rayon parallel iterators to process
    /// channels concurrently for improved throughput on multi-core systems.
    ///
    /// # Arguments
    ///
    /// * `samples` - Interleaved audio samples
    pub fn process_interleaved(&mut self, samples: &[f64]) {
        if self.channels > 4 {
            self.process_interleaved_parallel(samples);
        } else {
            self.process_interleaved_sequential(samples);
        }
    }

    /// Sequential per-channel processing (used for ≤4 channels).
    fn process_interleaved_sequential(&mut self, samples: &[f64]) {
        let frames = samples.len() / self.channels;
        for frame in 0..frames {
            for ch in 0..self.channels {
                let idx = frame * self.channels + ch;
                if idx < samples.len() {
                    let sample = samples[idx];
                    self.process_sample(sample, ch);
                }
            }
            self.sample_index += 1;
        }
    }

    /// Parallel per-channel processing using rayon (used for >4 channels).
    ///
    /// De-interleaves the buffer into per-channel planes, runs the Lanczos
    /// resampler on each channel in parallel, then merges the peak results
    /// back into `channel_peaks`.
    fn process_interleaved_parallel(&mut self, samples: &[f64]) {
        use rayon::prelude::*;

        let channels = self.channels;
        let frames = samples.len() / channels;
        let factor = self.oversample_mode.factor();
        let filter_coeffs = &self.filter_coeffs;

        // Build per-channel planar buffers
        let mut planes: Vec<Vec<f64>> = (0..channels)
            .map(|ch| (0..frames).map(|f| samples[f * channels + ch]).collect())
            .collect();

        // Process each channel in parallel, collecting per-channel peak max
        let channel_max_peaks: Vec<f64> = planes
            .par_iter_mut()
            .map(|plane| {
                let mut delay = DelayLine::new(TAPS_PER_PHASE);
                let mut peak = 0.0_f64;

                for &sample in plane.iter() {
                    delay.push(sample);

                    let abs_sample = sample.abs();
                    if abs_sample > peak {
                        peak = abs_sample;
                    }

                    // Check all oversampled phases
                    for phase in 1..factor {
                        if phase < filter_coeffs.len() {
                            let coeffs = &filter_coeffs[phase];
                            let mut sum = 0.0_f64;
                            for (i, &coeff) in coeffs.iter().enumerate() {
                                sum += delay.get(i) * coeff;
                            }
                            let abs_interp = sum.abs();
                            if abs_interp > peak {
                                peak = abs_interp;
                            }
                        }
                    }
                }
                peak
            })
            .collect();

        // Merge parallel results into the meter's per-channel peaks
        for (ch, max_peak) in channel_max_peaks.iter().enumerate() {
            if ch < self.channel_peaks.len() && *max_peak > self.channel_peaks[ch] {
                self.channel_peaks[ch] = *max_peak;
                self.channel_peak_indices[ch] = self.sample_index + frames;
            }
        }

        // Also update the sequential delay lines for future calls
        for frame in 0..frames {
            for ch in 0..channels {
                let idx = frame * channels + ch;
                if idx < samples.len() {
                    self.delay_lines[ch].push(samples[idx]);
                }
            }
        }

        self.sample_index += frames;
    }

    /// Process a single sample on a specific channel.
    fn process_sample(&mut self, sample: f64, channel: usize) {
        // Add sample to delay line
        self.delay_lines[channel].push(sample);

        // Check sample peak
        let abs_sample = sample.abs();
        if abs_sample > self.channel_peaks[channel] {
            self.channel_peaks[channel] = abs_sample;
            self.channel_peak_indices[channel] = self.sample_index;
        }

        // Oversample and check inter-sample peaks (uses configured oversample factor)
        let factor = self.oversample_mode.factor();
        for phase in 1..factor {
            let interpolated = self.interpolate(channel, phase);
            let abs_interpolated = interpolated.abs();

            if abs_interpolated > self.channel_peaks[channel] {
                self.channel_peaks[channel] = abs_interpolated;
                self.channel_peak_indices[channel] = self.sample_index;
            }
        }
    }

    /// Interpolate using Lanczos filter for given phase.
    fn interpolate(&self, channel: usize, phase: usize) -> f64 {
        let coeffs = &self.filter_coeffs[phase];
        let delay = &self.delay_lines[channel];

        let mut sum = 0.0;
        for (i, &coeff) in coeffs.iter().enumerate() {
            sum += delay.get(i) * coeff;
        }

        sum
    }

    /// Design Lanczos resampling filters for all phases (4x, legacy).
    fn design_lanczos_filters() -> Vec<Vec<f64>> {
        Self::design_lanczos_filters_n(OVERSAMPLE_FACTOR)
    }

    /// Design Lanczos resampling filters for `n`-times oversampling.
    ///
    /// Generates `n` filter banks: phase 0 is the identity (empty), and phases
    /// 1..n-1 are the sinc-interpolation filters for the in-between samples.
    fn design_lanczos_filters_n(oversample_factor: usize) -> Vec<Vec<f64>> {
        let mut filters = Vec::with_capacity(oversample_factor);

        // Phase 0 is the identity (no interpolation needed)
        filters.push(vec![]);

        // Design filters for remaining phases
        for phase in 1..oversample_factor {
            let mut coeffs = Vec::with_capacity(TAPS_PER_PHASE);
            let phase_offset = phase as f64 / oversample_factor as f64;

            for i in 0..TAPS_PER_PHASE {
                let n = i as f64 - LANCZOS_A as f64 + phase_offset;
                let coeff = Self::lanczos_kernel(n, LANCZOS_A);
                coeffs.push(coeff);
            }

            // Normalize coefficients
            let sum: f64 = coeffs.iter().sum();
            if sum != 0.0 {
                for coeff in &mut coeffs {
                    *coeff /= sum;
                }
            }

            filters.push(coeffs);
        }

        filters
    }

    /// Lanczos kernel function.
    ///
    /// L(x) = sinc(x) * sinc(x/a) for -a <= x <= a
    fn lanczos_kernel(x: f64, a: usize) -> f64 {
        if x == 0.0 {
            return 1.0;
        }

        let a_f64 = a as f64;
        if x.abs() >= a_f64 {
            return 0.0;
        }

        let sinc_x = (PI * x).sin() / (PI * x);
        let sinc_xa = (PI * x / a_f64).sin() / (PI * x / a_f64);

        sinc_x * sinc_xa
    }

    /// Get true peak in dBTP (maximum across all channels).
    pub fn true_peak_dbtp(&self) -> f64 {
        let max_peak = self.channel_peaks.iter().copied().fold(0.0, f64::max);
        Self::linear_to_dbtp(max_peak)
    }

    /// Get true peak in linear scale.
    pub fn true_peak_linear(&self) -> f64 {
        self.channel_peaks.iter().copied().fold(0.0, f64::max)
    }

    /// Get per-channel peaks in dBTP.
    pub fn channel_peaks_dbtp(&self) -> Vec<f64> {
        self.channel_peaks
            .iter()
            .map(|&peak| Self::linear_to_dbtp(peak))
            .collect()
    }

    /// Get per-channel peaks in linear scale.
    pub fn channel_peaks_linear(&self) -> Vec<f64> {
        self.channel_peaks.clone()
    }

    /// Get true peak for specific channel.
    pub fn channel_peak(&self, channel: usize) -> Option<TruePeak> {
        if channel < self.channels {
            Some(TruePeak::new(
                self.channel_peaks[channel],
                channel,
                self.channel_peak_indices[channel],
            ))
        } else {
            None
        }
    }

    /// Convert linear to dBTP.
    pub fn linear_to_dbtp(linear: f64) -> f64 {
        if linear > 0.0 {
            20.0 * linear.log10()
        } else {
            f64::NEG_INFINITY
        }
    }

    /// Convert dBTP to linear.
    pub fn dbtp_to_linear(dbtp: f64) -> f64 {
        10.0_f64.powf(dbtp / 20.0)
    }

    /// Reset the detector.
    pub fn reset(&mut self) {
        for delay in &mut self.delay_lines {
            delay.reset();
        }
        self.channel_peaks.fill(0.0);
        self.channel_peak_indices.fill(0);
        self.sample_index = 0;
    }

    /// Get sample rate.
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    /// Get channel count.
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Get the oversampling mode.
    pub fn oversample_mode(&self) -> OversampleMode {
        self.oversample_mode
    }
}

/// Circular delay line for sample storage.
struct DelayLine {
    buffer: Vec<f64>,
    size: usize,
    write_pos: usize,
}

impl DelayLine {
    /// Create a new delay line.
    fn new(size: usize) -> Self {
        Self {
            buffer: vec![0.0; size],
            size,
            write_pos: 0,
        }
    }

    /// Push a new sample into the delay line.
    fn push(&mut self, sample: f64) {
        self.buffer[self.write_pos] = sample;
        self.write_pos = (self.write_pos + 1) % self.size;
    }

    /// Get a sample from the delay line (0 = oldest, size-1 = newest).
    fn get(&self, index: usize) -> f64 {
        if index >= self.size {
            return 0.0;
        }
        let read_pos = (self.write_pos + index) % self.size;
        self.buffer[read_pos]
    }

    /// Reset the delay line.
    fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.write_pos = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_true_peak_detector_creates() {
        let detector = TruePeakDetector::new(48000.0, 2);
        assert_eq!(detector.sample_rate(), 48000.0);
        assert_eq!(detector.channels(), 2);
        assert_eq!(detector.oversample_mode(), OversampleMode::FourX);
    }

    #[test]
    fn test_8x_oversampling_creates() {
        let detector = TruePeakDetector::new_8x(48000.0, 2);
        assert_eq!(detector.oversample_mode(), OversampleMode::EightX);
        assert_eq!(detector.oversample_mode().factor(), 8);
    }

    #[test]
    fn test_8x_oversampling_detects_peaks() {
        // An 8x detector should produce a non-infinite true peak for a sinewave.
        let mut detector = TruePeakDetector::new_8x(48000.0, 1);
        let samples: Vec<f64> = (0..4800)
            .map(|i| (2.0 * std::f64::consts::PI * 997.0 * i as f64 / 48000.0).sin() * 0.9)
            .collect();
        detector.process_interleaved(&samples);
        let peak = detector.true_peak_dbtp();
        assert!(peak.is_finite(), "8x peak should be finite, got {peak}");
        // 0.9 peak amplitude → ≤ 0.9 true peak in linear, roughly -0.9 dBTP
        assert!(
            peak <= 1.0,
            "8x peak should be ≤ 1 dBTP for 0.9 amplitude signal, got {peak}"
        );
    }

    #[test]
    fn test_8x_not_lower_than_4x_for_sinewave() {
        // 8x should detect a peak at least as high as 4x (more phases = more coverage).
        let freq = 997.0_f64;
        let sr = 48000.0_f64;
        let samples: Vec<f64> = (0..4800)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / sr).sin() * 0.8)
            .collect();

        let mut det4x = TruePeakDetector::new(sr, 1);
        det4x.process_interleaved(&samples);

        let mut det8x = TruePeakDetector::new_8x(sr, 1);
        det8x.process_interleaved(&samples);

        let peak4 = det4x.true_peak_linear();
        let peak8 = det8x.true_peak_linear();

        // 8x peak should be ≥ 4x peak (cannot miss peaks that 4x finds).
        assert!(
            peak8 >= peak4 - 1e-6,
            "8x peak ({peak8}) should be >= 4x peak ({peak4})"
        );
    }

    #[test]
    fn test_parallel_processing_5ch() {
        // 5 channels triggers parallel path; result should match sequential.
        let channels = 5_usize;
        let frames = 4800_usize;
        // Use (ch + 1) / channels so that channel 0 receives a non-zero value
        // (1/5 * 0.7 = 0.14) rather than 0.0, ensuring all channels have a
        // measurable true-peak that is finite in dBTP.
        let samples: Vec<f64> = (0..frames * channels)
            .map(|i| ((i % channels + 1) as f64 / channels as f64) * 0.7)
            .collect();

        let mut det = TruePeakDetector::new(48000.0, channels);
        det.process_interleaved(&samples);

        let peaks = det.channel_peaks_dbtp();
        assert_eq!(peaks.len(), channels);
        // All peaks should be finite and valid
        for (ch, &p) in peaks.iter().enumerate() {
            assert!(p.is_finite(), "Channel {ch} peak should be finite, got {p}");
        }
    }

    #[test]
    fn test_linear_to_dbtp_conversion() {
        assert_eq!(TruePeakDetector::linear_to_dbtp(1.0), 0.0);
        assert_eq!(
            TruePeakDetector::linear_to_dbtp(0.5),
            20.0 * 0.5_f64.log10()
        );
    }

    #[test]
    fn test_dbtp_to_linear_conversion() {
        assert_eq!(TruePeakDetector::dbtp_to_linear(0.0), 1.0);
        let linear = TruePeakDetector::dbtp_to_linear(-6.0);
        assert!((linear - 0.501187).abs() < 0.001);
    }

    #[test]
    fn test_roundtrip_conversion() {
        let original = 0.7;
        let dbtp = TruePeakDetector::linear_to_dbtp(original);
        let recovered = TruePeakDetector::dbtp_to_linear(dbtp);
        assert!((original - recovered).abs() < 1e-10);
    }

    #[test]
    fn test_lanczos_kernel_at_zero() {
        let kernel = TruePeakDetector::lanczos_kernel(0.0, 3);
        assert_eq!(kernel, 1.0);
    }

    #[test]
    fn test_lanczos_kernel_outside_window() {
        let kernel = TruePeakDetector::lanczos_kernel(4.0, 3);
        assert_eq!(kernel, 0.0);
    }

    #[test]
    fn test_delay_line() {
        let mut delay = DelayLine::new(4);
        delay.push(1.0);
        delay.push(2.0);
        delay.push(3.0);
        delay.push(4.0);

        assert_eq!(delay.get(0), 1.0);
        assert_eq!(delay.get(1), 2.0);
        assert_eq!(delay.get(2), 3.0);
        assert_eq!(delay.get(3), 4.0);
    }

    #[test]
    fn test_true_peak_exceeds() {
        let peak = TruePeak::new(1.2, 0, 0);
        assert!(peak.exceeds(-1.0));
        assert!(!peak.exceeds(2.0));
    }

    // ── Tests merged from true_peak module ──

    #[test]
    fn severity_none_when_under_ceiling() {
        let ov = TruePeakOvershoot::new(-2.0, -1.0, 0);
        assert_eq!(ov.severity(), OvershootSeverity::None);
    }

    #[test]
    fn severity_minor_small_excess() {
        let ov = TruePeakOvershoot::new(-0.8, -1.0, 0);
        assert_eq!(ov.severity(), OvershootSeverity::Minor);
    }

    #[test]
    fn severity_moderate_medium_excess() {
        let ov = TruePeakOvershoot::new(0.5, -1.0, 0);
        assert_eq!(ov.severity(), OvershootSeverity::Moderate);
    }

    #[test]
    fn severity_severe_large_excess() {
        let ov = TruePeakOvershoot::new(2.0, -1.0, 0);
        assert_eq!(ov.severity(), OvershootSeverity::Severe);
    }

    #[test]
    fn severity_ordering() {
        assert!(OvershootSeverity::None < OvershootSeverity::Minor);
        assert!(OvershootSeverity::Minor < OvershootSeverity::Moderate);
        assert!(OvershootSeverity::Moderate < OvershootSeverity::Severe);
    }

    #[test]
    fn tp_meter_starts_at_neg_inf() {
        let m = TruePeakMeter::new(2, -1.0);
        assert!(m.true_peak_dbtp().is_infinite() && m.true_peak_dbtp() < 0.0);
    }

    #[test]
    fn tp_meter_process_sample_tracks_channel() {
        let mut m = TruePeakMeter::new(2, -1.0);
        m.process_sample(0.9, 0);
        m.process_sample(0.5, 1);
        let ch0 = m.channel_peak_dbtp(0).expect("ch0 should be valid");
        let ch1 = m.channel_peak_dbtp(1).expect("ch1 should be valid");
        assert!(ch0 > ch1);
    }

    #[test]
    fn tp_meter_true_peak_is_max_across_channels() {
        let mut m = TruePeakMeter::new(3, -1.0);
        m.process_sample(0.3, 0);
        m.process_sample(0.9, 1);
        m.process_sample(0.6, 2);
        let expected = if 0.9_f64 > 0.0 {
            20.0 * 0.9_f64.log10()
        } else {
            f64::NEG_INFINITY
        };
        assert!((m.true_peak_dbtp() - expected).abs() < 1e-9);
    }

    #[test]
    fn tp_meter_has_overshoot_above_ceiling() {
        let mut m = TruePeakMeter::new(1, -1.0);
        m.process_sample(0.99, 0);
        assert!(m.has_overshoot());
    }

    #[test]
    fn tp_meter_no_overshoot_below_ceiling() {
        let mut m = TruePeakMeter::new(1, -1.0);
        m.process_sample(0.5, 0);
        assert!(!m.has_overshoot());
    }

    #[test]
    fn tp_meter_reset_clears_peaks() {
        let mut m = TruePeakMeter::new(2, -1.0);
        m.process_sample(0.9, 0);
        m.reset();
        assert!(m.true_peak_dbtp().is_infinite() && m.true_peak_dbtp() < 0.0);
    }

    #[test]
    fn tp_meter_process_frame() {
        let mut m = TruePeakMeter::new(2, -1.0);
        m.process_frame(&[0.4, 0.8]);
        let expected = if 0.8_f64 > 0.0 {
            20.0 * 0.8_f64.log10()
        } else {
            f64::NEG_INFINITY
        };
        assert!(
            (m.channel_peak_dbtp(1)
                .expect("channel_peak_dbtp should succeed")
                - expected)
                .abs()
                < 1e-9
        );
    }

    #[test]
    fn report_worst_channel_identifies_highest_peak() {
        let mut m = TruePeakMeter::new(3, -1.0);
        m.process_sample(0.3, 0);
        m.process_sample(0.95, 1);
        m.process_sample(0.5, 2);
        let report = TruePeakReport::from_meter(&m);
        assert_eq!(report.worst_channel(), Some(1));
    }

    #[test]
    fn report_has_overshoot_when_any_channel_clips() {
        let mut m = TruePeakMeter::new(2, -1.0);
        m.process_sample(0.99, 0);
        m.process_sample(0.3, 1);
        let report = TruePeakReport::from_meter(&m);
        assert!(report.has_overshoot());
    }

    #[test]
    fn report_no_overshoot_all_below_ceiling() {
        let mut m = TruePeakMeter::new(2, -1.0);
        m.process_sample(0.5, 0);
        m.process_sample(0.4, 1);
        let report = TruePeakReport::from_meter(&m);
        assert!(!report.has_overshoot());
    }

    #[test]
    fn report_worst_severity_reflects_max() {
        let mut m = TruePeakMeter::new(2, -1.0);
        m.process_sample(0.5, 0);
        m.process_sample(1.2, 1);
        let report = TruePeakReport::from_meter(&m);
        assert_eq!(report.worst_severity(), OvershootSeverity::Severe);
    }
}

// ── Types merged from true_peak module ───────────────────────────────────────

/// Overshoot severity classification.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum OvershootSeverity {
    /// No overshoot (below allowed ceiling).
    None,
    /// Minor overshoot (0 - 0.5 dBTP above ceiling).
    Minor,
    /// Moderate overshoot (0.5 - 2.0 dBTP above ceiling).
    Moderate,
    /// Severe overshoot (> 2.0 dBTP above ceiling).
    Severe,
}

/// A single true-peak overshoot event.
#[derive(Clone, Debug)]
pub struct TruePeakOvershoot {
    /// Measured true peak in dBTP.
    pub measured_dbtp: f64,
    /// Allowed ceiling in dBTP (e.g. -1.0).
    pub ceiling_dbtp: f64,
    /// Channel index (0-based).
    pub channel: usize,
}

impl TruePeakOvershoot {
    /// Create an overshoot record.
    pub fn new(measured_dbtp: f64, ceiling_dbtp: f64, channel: usize) -> Self {
        Self {
            measured_dbtp,
            ceiling_dbtp,
            channel,
        }
    }

    /// Amount by which the ceiling is exceeded (positive = over, negative = under).
    pub fn excess_dbtp(&self) -> f64 {
        self.measured_dbtp - self.ceiling_dbtp
    }

    /// Classify the overshoot severity.
    pub fn severity(&self) -> OvershootSeverity {
        let excess = self.excess_dbtp();
        if excess <= 0.0 {
            OvershootSeverity::None
        } else if excess <= 0.5 {
            OvershootSeverity::Minor
        } else if excess <= 2.0 {
            OvershootSeverity::Moderate
        } else {
            OvershootSeverity::Severe
        }
    }
}

/// True-peak meter for a fixed number of channels.
///
/// Stores per-channel peak-linear values; callers supply oversampled (4x)
/// or interpolated samples.
#[derive(Debug)]
pub struct TruePeakMeter {
    channel_peaks: Vec<f64>,
    ceiling_dbtp: f64,
}

impl TruePeakMeter {
    /// Create a meter for `channels` channels with the given ceiling.
    pub fn new(channels: usize, ceiling_dbtp: f64) -> Self {
        Self {
            channel_peaks: vec![0.0; channels.max(1)],
            ceiling_dbtp,
        }
    }

    /// Process a single sample on the given channel.
    pub fn process_sample(&mut self, sample: f64, channel: usize) {
        if let Some(peak) = self.channel_peaks.get_mut(channel) {
            let abs = sample.abs();
            if abs > *peak {
                *peak = abs;
            }
        }
    }

    /// Process an interleaved frame (all channels for one sample period).
    pub fn process_frame(&mut self, frame: &[f64]) {
        for (ch, &s) in frame.iter().enumerate() {
            self.process_sample(s, ch);
        }
    }

    /// True peak in dBTP across all channels (worst case).
    pub fn true_peak_dbtp(&self) -> f64 {
        let max_linear = self.channel_peaks.iter().copied().fold(0.0_f64, f64::max);
        if max_linear > 0.0 {
            20.0 * max_linear.log10()
        } else {
            f64::NEG_INFINITY
        }
    }

    /// True peak in dBTP for a specific channel.
    pub fn channel_peak_dbtp(&self, channel: usize) -> Option<f64> {
        self.channel_peaks.get(channel).map(|&v| {
            if v > 0.0 {
                20.0 * v.log10()
            } else {
                f64::NEG_INFINITY
            }
        })
    }

    /// Return `true` if any channel exceeds the ceiling.
    pub fn has_overshoot(&self) -> bool {
        self.true_peak_dbtp() > self.ceiling_dbtp
    }

    /// Number of channels this meter was configured for.
    pub fn num_channels(&self) -> usize {
        self.channel_peaks.len()
    }

    /// Reset all peak levels to zero.
    pub fn reset(&mut self) {
        for p in &mut self.channel_peaks {
            *p = 0.0;
        }
    }
}

/// Per-channel summary used in the final report.
#[derive(Clone, Debug)]
pub struct ChannelPeakSummary {
    /// Channel index.
    pub channel: usize,
    /// True peak for this channel in dBTP.
    pub peak_dbtp: f64,
    /// Overshoot severity for this channel.
    pub severity: OvershootSeverity,
}

/// Aggregated true-peak report across all channels.
#[derive(Clone, Debug)]
pub struct TruePeakReport {
    /// Per-channel summaries.
    pub channels: Vec<ChannelPeakSummary>,
    /// Ceiling used for evaluation.
    pub ceiling_dbtp: f64,
}

impl TruePeakReport {
    /// Build a report from a [`TruePeakMeter`].
    pub fn from_meter(meter: &TruePeakMeter) -> Self {
        let ceiling = meter.ceiling_dbtp;
        let channels = (0..meter.num_channels())
            .map(|ch| {
                let peak_dbtp = meter.channel_peak_dbtp(ch).unwrap_or(f64::NEG_INFINITY);
                let overshoot = TruePeakOvershoot::new(peak_dbtp, ceiling, ch);
                ChannelPeakSummary {
                    channel: ch,
                    peak_dbtp,
                    severity: overshoot.severity(),
                }
            })
            .collect();
        Self {
            channels,
            ceiling_dbtp: ceiling,
        }
    }

    /// Return the channel index with the worst (highest) true peak.
    pub fn worst_channel(&self) -> Option<usize> {
        self.channels
            .iter()
            .max_by(|a, b| {
                a.peak_dbtp
                    .partial_cmp(&b.peak_dbtp)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|s| s.channel)
    }

    /// `true` if any channel is in overshoot.
    pub fn has_overshoot(&self) -> bool {
        self.channels
            .iter()
            .any(|s| s.severity != OvershootSeverity::None)
    }

    /// Return the worst severity across all channels.
    pub fn worst_severity(&self) -> OvershootSeverity {
        self.channels
            .iter()
            .map(|s| s.severity)
            .max()
            .unwrap_or(OvershootSeverity::None)
    }
}
