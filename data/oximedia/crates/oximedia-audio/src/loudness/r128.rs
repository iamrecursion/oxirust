//! EBU R128 loudness measurement implementation.
//!
//! Implements the complete EBU R128 / ITU-R BS.1770-4 loudness measurement
//! algorithm with momentary, short-term, and integrated loudness.

#![forbid(unsafe_code)]
#![allow(clippy::cast_lossless)]

use super::filter::KWeightFilterBank;
use super::gate::{BlockAccumulator, GatingProcessor};
use super::peak::TruePeakDetector;

/// Momentary loudness window duration in milliseconds.
const MOMENTARY_WINDOW_MS: f64 = 400.0;

/// Short-term loudness window duration in milliseconds.
const SHORT_TERM_WINDOW_MS: f64 = 3000.0;

/// Overlap between successive measurement blocks (75% overlap).
const BLOCK_OVERLAP: f64 = 0.75;

/// EBU R128 loudness meter.
///
/// Provides comprehensive loudness measurement including:
/// - Momentary loudness (400ms)
/// - Short-term loudness (3s)
/// - Integrated loudness (entire program)
/// - Loudness range (LRA)
/// - True peak level
#[derive(Clone, Debug)]
pub struct R128Meter {
    /// Sample rate in Hz.
    sample_rate: f64,
    /// Number of audio channels.
    channels: usize,
    /// K-weighting filter bank.
    filter_bank: KWeightFilterBank,
    /// Gating processor for integrated loudness.
    gating: GatingProcessor,
    /// Momentary loudness window (400ms with overlap).
    momentary_window: SlidingWindow,
    /// Short-term loudness window (3s with overlap).
    short_term_window: SlidingWindow,
    /// Block accumulator for integrated loudness.
    integrated_blocks: BlockAccumulator,
    /// True peak detector.
    peak_detector: TruePeakDetector,
    /// Filtered sample buffer for block processing.
    filtered_buffer: Vec<f64>,
    /// Current momentary loudness in LUFS.
    momentary_loudness: f64,
    /// Current short-term loudness in LUFS.
    short_term_loudness: f64,
    /// Maximum momentary loudness seen.
    max_momentary: f64,
    /// Maximum short-term loudness seen.
    max_short_term: f64,
    /// True peak value (linear).
    true_peak: f64,
}

impl R128Meter {
    /// Create a new R128 loudness meter.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz
    /// * `channels` - Number of audio channels
    #[must_use]
    pub fn new(sample_rate: f64, channels: usize) -> Self {
        let filter_bank = KWeightFilterBank::new(channels, sample_rate);
        let gating = GatingProcessor::new(sample_rate, channels);

        let momentary_samples = (sample_rate * MOMENTARY_WINDOW_MS / 1000.0) as usize * channels;
        let short_term_samples = (sample_rate * SHORT_TERM_WINDOW_MS / 1000.0) as usize * channels;

        let momentary_window = SlidingWindow::new(momentary_samples, BLOCK_OVERLAP);
        let short_term_window = SlidingWindow::new(short_term_samples, BLOCK_OVERLAP);

        let integrated_blocks = BlockAccumulator::new(sample_rate, channels, MOMENTARY_WINDOW_MS);
        let peak_detector = TruePeakDetector::new(sample_rate, channels);

        Self {
            sample_rate,
            channels,
            filter_bank,
            gating,
            momentary_window,
            short_term_window,
            integrated_blocks,
            peak_detector,
            filtered_buffer: Vec::new(),
            momentary_loudness: f64::NEG_INFINITY,
            short_term_loudness: f64::NEG_INFINITY,
            max_momentary: f64::NEG_INFINITY,
            max_short_term: f64::NEG_INFINITY,
            true_peak: 0.0,
        }
    }

    /// Process a block of interleaved audio samples.
    ///
    /// # Arguments
    ///
    /// * `samples` - Interleaved audio samples (normalized -1.0 to 1.0)
    pub fn process_interleaved(&mut self, samples: &[f64]) {
        if samples.is_empty() {
            return;
        }

        // Apply K-weighting filter
        self.filtered_buffer.resize(samples.len(), 0.0);
        self.filter_bank
            .process_interleaved(samples, self.channels, &mut self.filtered_buffer);

        // Update true peak (on original unfiltered samples)
        let peak = self.peak_detector.process_interleaved(samples);
        self.true_peak = self.true_peak.max(peak);

        // Process momentary loudness (400ms window)
        if let Some(window_samples) = self.momentary_window.add_samples(&self.filtered_buffer) {
            let power = self.gating.calculate_block_power(window_samples);
            self.momentary_loudness = GatingProcessor::power_to_lufs(power);
            self.max_momentary = self.max_momentary.max(self.momentary_loudness);
        }

        // Process short-term loudness (3s window)
        if let Some(window_samples) = self.short_term_window.add_samples(&self.filtered_buffer) {
            let power = self.gating.calculate_block_power(window_samples);
            self.short_term_loudness = GatingProcessor::power_to_lufs(power);
            self.max_short_term = self.max_short_term.max(self.short_term_loudness);
        }

        // Accumulate for integrated loudness
        self.integrated_blocks.add_samples(&self.filtered_buffer);
    }

    /// Process planar audio samples.
    ///
    /// # Arguments
    ///
    /// * `channels` - Mutable slice of per-channel sample buffers
    pub fn process_planar(&mut self, channels: &mut [Vec<f64>]) {
        if channels.is_empty() {
            return;
        }

        // Apply K-weighting filter
        self.filter_bank.process_planar(channels);

        // Convert to interleaved for processing
        let num_channels = channels.len();
        let num_frames = channels[0].len();
        let mut interleaved = vec![0.0; num_frames * num_channels];

        for frame in 0..num_frames {
            for (ch_idx, ch_samples) in channels.iter().enumerate() {
                interleaved[frame * num_channels + ch_idx] = ch_samples[frame];
            }
        }

        // Process interleaved
        self.process_interleaved(&interleaved);
    }

    /// Get the current momentary loudness in LUFS.
    ///
    /// Momentary loudness uses a 400ms sliding window with 75% overlap.
    #[must_use]
    pub fn momentary_loudness(&self) -> f64 {
        self.momentary_loudness
    }

    /// Get the current short-term loudness in LUFS.
    ///
    /// Short-term loudness uses a 3-second sliding window with 75% overlap.
    #[must_use]
    pub fn short_term_loudness(&self) -> f64 {
        self.short_term_loudness
    }

    /// Get the integrated loudness in LUFS.
    ///
    /// Integrated loudness is the gated mean loudness over the entire program.
    #[must_use]
    pub fn integrated_loudness(&self) -> f64 {
        self.integrated_blocks.integrated_loudness()
    }

    /// Get the loudness range in LU.
    ///
    /// LRA is the difference between the 95th and 10th percentile
    /// of the short-term loudness distribution.
    #[must_use]
    pub fn loudness_range(&self) -> f64 {
        self.integrated_blocks.loudness_range()
    }

    /// Get the maximum momentary loudness seen in LUFS.
    #[must_use]
    pub fn max_momentary(&self) -> f64 {
        self.max_momentary
    }

    /// Get the maximum short-term loudness seen in LUFS.
    #[must_use]
    pub fn max_short_term(&self) -> f64 {
        self.max_short_term
    }

    /// Get the true peak level in dBTP.
    #[must_use]
    pub fn true_peak_dbtp(&self) -> f64 {
        TruePeakDetector::linear_to_dbtp(self.true_peak)
    }

    /// Get the true peak level (linear).
    #[must_use]
    pub fn true_peak_linear(&self) -> f64 {
        self.true_peak
    }

    /// Get per-channel true peak levels.
    #[must_use]
    pub fn channel_peaks(&self) -> Vec<f64> {
        self.peak_detector.get_all_peaks()
    }

    /// Reset the meter.
    pub fn reset(&mut self) {
        self.filter_bank.reset();
        self.momentary_window.reset();
        self.short_term_window.reset();
        self.integrated_blocks.reset();
        self.peak_detector.reset();
        self.filtered_buffer.clear();
        self.momentary_loudness = f64::NEG_INFINITY;
        self.short_term_loudness = f64::NEG_INFINITY;
        self.max_momentary = f64::NEG_INFINITY;
        self.max_short_term = f64::NEG_INFINITY;
        self.true_peak = 0.0;
    }

    /// Get the sample rate.
    #[must_use]
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    /// Get the number of channels.
    #[must_use]
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Check if integrated loudness measurement is valid.
    ///
    /// Requires at least a minimum amount of audio to be processed.
    #[must_use]
    pub fn has_valid_integrated(&self) -> bool {
        self.integrated_blocks.block_count() > 0 && !self.integrated_loudness().is_infinite()
    }

    /// Get the number of samples processed.
    #[must_use]
    pub fn samples_processed(&self) -> usize {
        self.momentary_window.total_samples_added() / self.channels
    }
}

/// Sliding window for momentary/short-term loudness measurement.
///
/// Maintains a circular buffer with configurable overlap.
#[derive(Clone, Debug)]
struct SlidingWindow {
    /// Window buffer.
    buffer: Vec<f64>,
    /// Window size in samples.
    window_size: usize,
    /// Hop size (how much to advance on each step).
    hop_size: usize,
    /// Current position in buffer.
    position: usize,
    /// Number of samples added since last window output.
    samples_since_hop: usize,
    /// Total samples added (for statistics).
    total_added: usize,
}

impl SlidingWindow {
    /// Create a new sliding window.
    ///
    /// # Arguments
    ///
    /// * `window_size` - Size of the window in samples
    /// * `overlap` - Overlap ratio (0.0 to 1.0, e.g., 0.75 for 75% overlap)
    fn new(window_size: usize, overlap: f64) -> Self {
        let hop_size = ((1.0 - overlap) * window_size as f64) as usize;

        Self {
            buffer: vec![0.0; window_size],
            window_size,
            hop_size: hop_size.max(1),
            position: 0,
            samples_since_hop: 0,
            total_added: 0,
        }
    }

    /// Add samples to the window.
    ///
    /// # Arguments
    ///
    /// * `samples` - Samples to add
    ///
    /// # Returns
    ///
    /// Some(window_samples) when a new window is ready, None otherwise
    fn add_samples(&mut self, samples: &[f64]) -> Option<&[f64]> {
        let mut window_ready = false;

        for &sample in samples {
            self.buffer[self.position] = sample;
            self.position = (self.position + 1) % self.window_size;
            self.samples_since_hop += 1;
            self.total_added += 1;

            // Time for a new window?
            if self.samples_since_hop >= self.hop_size && self.total_added >= self.window_size {
                window_ready = true;
                self.samples_since_hop = 0;
            }
        }

        if window_ready {
            Some(self.get_window())
        } else {
            None
        }
    }

    /// Get the current window contents in order.
    fn get_window(&self) -> &[f64] {
        &self.buffer
    }

    /// Reset the window.
    fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.position = 0;
        self.samples_since_hop = 0;
        self.total_added = 0;
    }

    /// Get total samples added.
    fn total_samples_added(&self) -> usize {
        self.total_added
    }
}

/// ATSC A/85 loudness meter.
///
/// ATSC A/85 uses the same measurement algorithm as EBU R128
/// but with different terminology (LKFS instead of LUFS).
/// This is essentially an alias for R128Meter with ATSC-specific methods.
pub type AtscA85Meter = R128Meter;

/// Extension trait for ATSC A/85 specific terminology.
pub trait AtscA85Ext {
    /// Get integrated LKFS (Loudness, K-weighted, relative to Full Scale).
    ///
    /// This is identical to integrated LUFS.
    fn integrated_lkfs(&self) -> f64;

    /// Get momentary LKFS.
    fn momentary_lkfs(&self) -> f64;

    /// Get short-term LKFS.
    fn short_term_lkfs(&self) -> f64;

    /// Check ATSC A/85 compliance.
    ///
    /// ATSC A/85 recommends -24 LKFS ±2 dB for most content.
    fn check_atsc_compliance(&self) -> ComplianceStatus;
}

impl AtscA85Ext for AtscA85Meter {
    fn integrated_lkfs(&self) -> f64 {
        self.integrated_loudness()
    }

    fn momentary_lkfs(&self) -> f64 {
        self.momentary_loudness()
    }

    fn short_term_lkfs(&self) -> f64 {
        self.short_term_loudness()
    }

    fn check_atsc_compliance(&self) -> ComplianceStatus {
        let target = -24.0;
        let tolerance = 2.0;
        let lkfs = self.integrated_lkfs();

        if lkfs.is_infinite() {
            ComplianceStatus::Unknown
        } else if lkfs >= target - tolerance && lkfs <= target + tolerance {
            ComplianceStatus::Compliant
        } else if lkfs > target + tolerance {
            ComplianceStatus::TooLoud(lkfs - target)
        } else {
            ComplianceStatus::TooQuiet(target - lkfs)
        }
    }
}

/// Compliance status for broadcast standards.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ComplianceStatus {
    /// Compliant with standard.
    Compliant,
    /// Too loud by specified amount in dB.
    TooLoud(f64),
    /// Too quiet by specified amount in dB.
    TooQuiet(f64),
    /// Unknown (not enough data).
    Unknown,
}

impl ComplianceStatus {
    /// Check if compliant.
    #[must_use]
    pub fn is_compliant(&self) -> bool {
        matches!(self, Self::Compliant)
    }

    /// Get deviation from target in dB (positive = too loud, negative = too quiet).
    #[must_use]
    pub fn deviation(&self) -> Option<f64> {
        match self {
            Self::TooLoud(db) => Some(*db),
            Self::TooQuiet(db) => Some(-*db),
            Self::Compliant => Some(0.0),
            Self::Unknown => None,
        }
    }
}

/// EBU R128 compliance checker.
pub struct R128Compliance;

impl R128Compliance {
    /// Check EBU R128 program loudness compliance.
    ///
    /// EBU R128 recommends -23 LUFS ±1 LU for program loudness.
    ///
    /// # Arguments
    ///
    /// * `integrated_lufs` - Integrated loudness in LUFS
    #[must_use]
    pub fn check_program_loudness(integrated_lufs: f64) -> ComplianceStatus {
        let target = -23.0;
        let tolerance = 1.0;

        if integrated_lufs.is_infinite() {
            ComplianceStatus::Unknown
        } else if integrated_lufs >= target - tolerance && integrated_lufs <= target + tolerance {
            ComplianceStatus::Compliant
        } else if integrated_lufs > target + tolerance {
            ComplianceStatus::TooLoud(integrated_lufs - target)
        } else {
            ComplianceStatus::TooQuiet(target - integrated_lufs)
        }
    }

    /// Check EBU R128 true peak compliance.
    ///
    /// EBU R128 recommends maximum true peak of -1 dBTP.
    ///
    /// # Arguments
    ///
    /// * `true_peak_dbtp` - True peak in dBTP
    #[must_use]
    pub fn check_true_peak(true_peak_dbtp: f64) -> bool {
        true_peak_dbtp <= -1.0
    }

    /// Check EBU R128 loudness range.
    ///
    /// No strict requirement, but typically expect 5-20 LU for most content.
    ///
    /// # Arguments
    ///
    /// * `lra` - Loudness range in LU
    #[must_use]
    pub fn check_loudness_range(lra: f64) -> bool {
        lra >= 1.0 && lra <= 30.0
    }

    /// Get recommended gain adjustment to meet target loudness.
    ///
    /// # Arguments
    ///
    /// * `measured_lufs` - Measured integrated loudness in LUFS
    /// * `target_lufs` - Target loudness in LUFS
    ///
    /// # Returns
    ///
    /// Gain adjustment in dB (positive = increase gain, negative = decrease gain)
    #[must_use]
    pub fn recommended_gain_adjustment(measured_lufs: f64, target_lufs: f64) -> f64 {
        if measured_lufs.is_infinite() {
            0.0
        } else {
            target_lufs - measured_lufs
        }
    }
}
