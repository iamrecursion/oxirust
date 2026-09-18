#![allow(dead_code)]
//! Silence detection and dead-air monitoring for broadcast audio.
//!
//! Detects periods of silence or near-silence in audio streams, essential
//! for broadcast compliance monitoring and dead-air prevention. Supports
//! configurable threshold, minimum duration, and per-channel operation.
//!
//! # Relationship to the main metering pipeline
//!
//! This module is independent of the [`crate::ebu_r128_impl::EbuR128Meter`]
//! filter → LKFS → gating → LRA pipeline (see the crate-level docs, "Pipeline
//! Architecture") — it shares no state with it. It is easily confused with
//! that pipeline's −70 LUFS absolute gate ([`crate::ebu_r128_impl`]'s
//! `ABSOLUTE_GATE`, used by both `integrated_lufs()` and `loudness_range_lu()`),
//! but the two serve different purposes: the absolute gate silently *excludes*
//! low-level 400 ms blocks from the loudness statistics, while this module
//! *reports* sustained silence as a discrete, timestamped dead-air event. Run
//! it alongside [`crate::LoudnessMeter`] on the same audio buffer. See also
//! [`crate::clip_counter`] and [`crate::rms_envelope`] for the other two
//! complementary, independently-run QC modules.

/// Configuration for silence detection.
#[derive(Clone, Debug)]
pub struct SilenceDetectConfig {
    /// Sample rate in Hz.
    pub sample_rate: f64,
    /// Number of audio channels.
    pub channels: usize,
    /// Threshold in dBFS below which audio is considered silent.
    pub threshold_dbfs: f64,
    /// Minimum duration in seconds for a silence event to be reported.
    pub min_duration: f64,
    /// Whether to use RMS measurement (true) or peak measurement (false).
    pub use_rms: bool,
    /// RMS measurement window in samples (only used if `use_rms` is true).
    pub rms_window: usize,
}

impl SilenceDetectConfig {
    /// Create a new silence detection configuration.
    pub fn new(sample_rate: f64, channels: usize) -> Self {
        Self {
            sample_rate,
            channels: channels.max(1),
            threshold_dbfs: -60.0,
            min_duration: 1.0,
            use_rms: true,
            rms_window: 1024,
        }
    }

    /// Set the silence threshold in dBFS.
    pub fn with_threshold(mut self, dbfs: f64) -> Self {
        self.threshold_dbfs = dbfs;
        self
    }

    /// Set the minimum silence duration in seconds.
    pub fn with_min_duration(mut self, seconds: f64) -> Self {
        self.min_duration = seconds.max(0.0);
        self
    }

    /// Use RMS measurement for level detection.
    pub fn with_rms(mut self, use_rms: bool) -> Self {
        self.use_rms = use_rms;
        self
    }

    /// Set the RMS window size.
    pub fn with_rms_window(mut self, size: usize) -> Self {
        self.rms_window = size.max(1);
        self
    }

    /// Convert the threshold to linear scale.
    fn threshold_linear(&self) -> f64 {
        10.0_f64.powf(self.threshold_dbfs / 20.0)
    }

    /// Minimum duration in samples.
    fn min_duration_samples(&self) -> u64 {
        (self.min_duration * self.sample_rate).ceil() as u64
    }
}

impl Default for SilenceDetectConfig {
    fn default() -> Self {
        Self::new(48000.0, 2)
    }
}

/// A detected silence event.
#[derive(Clone, Debug, PartialEq)]
pub struct SilenceEvent {
    /// Sample offset where the silence started.
    pub start_sample: u64,
    /// Duration of the silence in samples.
    pub duration_samples: u64,
    /// Average level during the silence (linear scale).
    pub avg_level: f64,
}

impl SilenceEvent {
    /// Duration in seconds.
    #[allow(clippy::cast_precision_loss)]
    pub fn duration_seconds(&self, sample_rate: f64) -> f64 {
        self.duration_samples as f64 / sample_rate
    }

    /// Start time in seconds.
    #[allow(clippy::cast_precision_loss)]
    pub fn start_seconds(&self, sample_rate: f64) -> f64 {
        self.start_sample as f64 / sample_rate
    }
}

/// Per-channel silence tracking state.
#[derive(Clone, Debug)]
struct ChannelSilenceState {
    /// Whether we are currently in a silence region.
    in_silence: bool,
    /// Sample offset where the current silence started.
    silence_start: u64,
    /// Duration of the current silence in samples.
    silence_duration: u64,
    /// Sum of levels during current silence (for averaging).
    level_sum: f64,
    /// Ring buffer for RMS calculation.
    rms_buffer: Vec<f64>,
    /// Write position in the ring buffer.
    rms_write_pos: usize,
    /// Running sum of squares.
    rms_sum_sq: f64,
    /// Valid samples in the RMS buffer.
    rms_valid: usize,
    /// Total samples processed.
    samples_processed: u64,
}

impl ChannelSilenceState {
    /// Create a new state.
    fn new(rms_window: usize) -> Self {
        Self {
            in_silence: false,
            silence_start: 0,
            silence_duration: 0,
            level_sum: 0.0,
            rms_buffer: vec![0.0; rms_window],
            rms_write_pos: 0,
            rms_sum_sq: 0.0,
            rms_valid: 0,
            samples_processed: 0,
        }
    }

    /// Reset to initial state.
    fn reset(&mut self) {
        self.in_silence = false;
        self.silence_start = 0;
        self.silence_duration = 0;
        self.level_sum = 0.0;
        self.rms_sum_sq = 0.0;
        self.rms_valid = 0;
        self.rms_write_pos = 0;
        self.samples_processed = 0;
        for v in &mut self.rms_buffer {
            *v = 0.0;
        }
    }
}

/// Silence detector for broadcast audio monitoring.
#[derive(Clone, Debug)]
pub struct SilenceDetector {
    /// Configuration.
    config: SilenceDetectConfig,
    /// Linear threshold.
    threshold_linear: f64,
    /// Minimum duration in samples.
    min_duration_samples: u64,
    /// Per-channel state.
    channel_states: Vec<ChannelSilenceState>,
    /// Detected silence events.
    events: Vec<SilenceEvent>,
    /// Maximum events to store.
    max_events: usize,
    /// Whether silence is currently active (any channel).
    is_silent: bool,
    /// Total silence duration in samples across all events.
    total_silence_samples: u64,
}

impl SilenceDetector {
    /// Create a new silence detector.
    pub fn new(config: SilenceDetectConfig) -> Self {
        let threshold_linear = config.threshold_linear();
        let min_duration_samples = config.min_duration_samples();
        let channels = config.channels;
        let rms_window = config.rms_window;
        Self {
            config,
            threshold_linear,
            min_duration_samples,
            channel_states: (0..channels)
                .map(|_| ChannelSilenceState::new(rms_window))
                .collect(),
            events: Vec::new(),
            max_events: 10_000,
            is_silent: false,
            total_silence_samples: 0,
        }
    }

    /// Process interleaved audio samples.
    pub fn process_interleaved(&mut self, samples: &[f64]) {
        let frame_count = samples.len() / self.config.channels;
        for frame in 0..frame_count {
            let mut all_silent = true;
            for ch in 0..self.config.channels {
                let sample = samples[frame * self.config.channels + ch];
                let is_ch_silent = self.process_channel_sample(ch, sample);
                if !is_ch_silent {
                    all_silent = false;
                }
            }
            self.is_silent = all_silent;
        }
    }

    /// Process a single sample for a channel, returns whether the channel is currently silent.
    #[allow(clippy::cast_precision_loss)]
    fn process_channel_sample(&mut self, channel: usize, sample: f64) -> bool {
        let state = &mut self.channel_states[channel];
        let sq = sample * sample;

        // Update RMS buffer
        let window_len = state.rms_buffer.len();
        let old = state.rms_buffer[state.rms_write_pos];
        state.rms_sum_sq += sq - old;
        if state.rms_sum_sq < 0.0 {
            state.rms_sum_sq = 0.0;
        }
        state.rms_buffer[state.rms_write_pos] = sq;
        state.rms_write_pos = (state.rms_write_pos + 1) % window_len;
        if state.rms_valid < window_len {
            state.rms_valid += 1;
        }

        // Compute current level
        let level = if self.config.use_rms && state.rms_valid > 0 {
            (state.rms_sum_sq / state.rms_valid as f64).sqrt()
        } else {
            sample.abs()
        };

        let is_below = level < self.threshold_linear;

        if is_below {
            if !state.in_silence {
                state.in_silence = true;
                state.silence_start = state.samples_processed;
                state.silence_duration = 1;
                state.level_sum = level;
            } else {
                state.silence_duration += 1;
                state.level_sum += level;
            }
        } else if state.in_silence {
            // Silence ended
            if state.silence_duration >= self.min_duration_samples
                && self.events.len() < self.max_events
            {
                let avg = if state.silence_duration > 0 {
                    state.level_sum / state.silence_duration as f64
                } else {
                    0.0
                };
                self.events.push(SilenceEvent {
                    start_sample: state.silence_start,
                    duration_samples: state.silence_duration,
                    avg_level: avg,
                });
                self.total_silence_samples += state.silence_duration;
            }
            state.in_silence = false;
            state.silence_duration = 0;
            state.level_sum = 0.0;
        }

        state.samples_processed += 1;
        is_below
    }

    /// Check if all channels are currently silent.
    pub fn is_silent(&self) -> bool {
        self.is_silent
    }

    /// Get the number of detected silence events.
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Get all detected silence events.
    pub fn events(&self) -> &[SilenceEvent] {
        &self.events
    }

    /// Get the total silence duration in seconds.
    #[allow(clippy::cast_precision_loss)]
    pub fn total_silence_seconds(&self) -> f64 {
        self.total_silence_samples as f64 / self.config.sample_rate
    }

    /// Get total samples processed (per channel).
    pub fn total_samples(&self) -> u64 {
        self.channel_states
            .first()
            .map_or(0, |s| s.samples_processed)
    }

    /// Get the percentage of audio that was silence.
    #[allow(clippy::cast_precision_loss)]
    pub fn silence_percentage(&self) -> f64 {
        let total = self.total_samples();
        if total == 0 {
            return 0.0;
        }
        self.total_silence_samples as f64 / total as f64 * 100.0
    }

    /// Reset the detector.
    pub fn reset(&mut self) {
        for state in &mut self.channel_states {
            state.reset();
        }
        self.events.clear();
        self.is_silent = false;
        self.total_silence_samples = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_detector() -> SilenceDetector {
        let config = SilenceDetectConfig::new(48000.0, 1)
            .with_threshold(-40.0)
            .with_min_duration(0.01) // 480 samples at 48kHz
            .with_rms(false); // Use peak for simpler testing
        SilenceDetector::new(config)
    }

    #[test]
    fn test_config_defaults() {
        let config = SilenceDetectConfig::new(48000.0, 2);
        assert!((config.threshold_dbfs - (-60.0)).abs() < 1e-12);
        assert!((config.min_duration - 1.0).abs() < 1e-12);
        assert!(config.use_rms);
    }

    #[test]
    fn test_config_builder() {
        let config = SilenceDetectConfig::new(44100.0, 1)
            .with_threshold(-50.0)
            .with_min_duration(0.5)
            .with_rms(false)
            .with_rms_window(2048);
        assert!((config.threshold_dbfs - (-50.0)).abs() < 1e-12);
        assert!((config.min_duration - 0.5).abs() < 1e-12);
        assert!(!config.use_rms);
        assert_eq!(config.rms_window, 2048);
    }

    #[test]
    fn test_no_silence_loud_signal() {
        let mut det = make_detector();
        let samples = vec![0.5; 2000];
        det.process_interleaved(&samples);
        assert_eq!(det.event_count(), 0);
        assert!(!det.is_silent());
    }

    #[test]
    fn test_detect_silence_region() {
        let mut det = make_detector();
        // Loud -> silent -> loud
        let mut samples = vec![0.5; 1000];
        samples.extend(vec![0.0001; 1000]); // below -40 dBFS
        samples.extend(vec![0.5; 1000]);
        det.process_interleaved(&samples);
        assert!(det.event_count() >= 1);
    }

    #[test]
    fn test_short_silence_ignored() {
        let config = SilenceDetectConfig::new(48000.0, 1)
            .with_threshold(-40.0)
            .with_min_duration(1.0) // 1 second minimum
            .with_rms(false);
        let mut det = SilenceDetector::new(config);
        // Only 100 silent samples (much less than 1 second)
        let mut samples = vec![0.5; 500];
        samples.extend(vec![0.0001; 100]);
        samples.extend(vec![0.5; 500]);
        det.process_interleaved(&samples);
        assert_eq!(det.event_count(), 0);
    }

    #[test]
    fn test_silence_event_timing() {
        let mut det = make_detector();
        let mut samples = vec![0.5; 500];
        let silence_start = samples.len();
        samples.extend(vec![0.0001; 1000]);
        samples.extend(vec![0.5; 500]);
        det.process_interleaved(&samples);
        assert!(det.event_count() >= 1);
        let evt = &det.events()[0];
        assert_eq!(evt.start_sample, silence_start as u64);
    }

    #[test]
    fn test_silence_event_duration() {
        let mut det = make_detector();
        let mut samples = vec![0.5; 500];
        samples.extend(vec![0.0001; 1000]);
        samples.extend(vec![0.5; 500]);
        det.process_interleaved(&samples);
        assert!(det.event_count() >= 1);
        let evt = &det.events()[0];
        assert_eq!(evt.duration_samples, 1000);
        let dur = evt.duration_seconds(48000.0);
        assert!((dur - 1000.0 / 48000.0).abs() < 1e-6);
    }

    #[test]
    fn test_stereo_silence() {
        let config = SilenceDetectConfig::new(48000.0, 2)
            .with_threshold(-40.0)
            .with_min_duration(0.01)
            .with_rms(false);
        let mut det = SilenceDetector::new(config);
        // Interleaved stereo: both channels silent
        let mut samples = Vec::new();
        for _ in 0..500 {
            samples.push(0.5);
            samples.push(0.5);
        }
        for _ in 0..1000 {
            samples.push(0.0001);
            samples.push(0.0001);
        }
        for _ in 0..500 {
            samples.push(0.5);
            samples.push(0.5);
        }
        det.process_interleaved(&samples);
        assert!(det.event_count() >= 1);
    }

    #[test]
    fn test_total_silence_seconds() {
        let mut det = make_detector();
        let mut samples = vec![0.5; 500];
        samples.extend(vec![0.0001; 48000]); // 1 second of silence
        samples.extend(vec![0.5; 500]);
        det.process_interleaved(&samples);
        let secs = det.total_silence_seconds();
        assert!((secs - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_silence_percentage() {
        let mut det = make_detector();
        // Half silence, half not (using peak mode)
        let mut samples = vec![0.5; 1000];
        samples.extend(vec![0.0001; 1000]);
        samples.extend(vec![0.5; 100]); // end with loud to finalize event
        det.process_interleaved(&samples);
        let pct = det.silence_percentage();
        // ~1000/2100 ~ 47.6%
        assert!(pct > 40.0 && pct < 55.0, "Expected ~47%, got {pct}");
    }

    #[test]
    fn test_reset() {
        let mut det = make_detector();
        det.process_interleaved(&vec![0.0001; 2000]);
        det.process_interleaved(&vec![0.5; 100]); // finalize event
        assert!(det.event_count() > 0);
        det.reset();
        assert_eq!(det.event_count(), 0);
        assert!(!det.is_silent());
    }

    #[test]
    fn test_silence_event_start_seconds() {
        let evt = SilenceEvent {
            start_sample: 48000,
            duration_samples: 24000,
            avg_level: 0.0001,
        };
        assert!((evt.start_seconds(48000.0) - 1.0).abs() < 1e-12);
        assert!((evt.duration_seconds(48000.0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_threshold_conversion() {
        let config = SilenceDetectConfig::new(48000.0, 1).with_threshold(-20.0);
        let linear = config.threshold_linear();
        // -20 dBFS = 0.1
        assert!((linear - 0.1).abs() < 0.001);
    }
}
