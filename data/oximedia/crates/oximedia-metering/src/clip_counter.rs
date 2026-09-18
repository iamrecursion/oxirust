#![allow(dead_code)]
//! Audio clip/overload event counting and tracking.
//!
//! Monitors audio signals for clipping events (samples at or exceeding 0 dBFS),
//! counts consecutive clip samples, logs clip event timestamps, and provides
//! statistics for broadcast quality control. Supports per-channel tracking
//! and configurable clip thresholds.
//!
//! # Relationship to the main metering pipeline
//!
//! This module is independent of the [`crate::ebu_r128_impl::EbuR128Meter`]
//! filter → LKFS → gating → LRA pipeline (see the crate-level docs, "Pipeline
//! Architecture") — it shares no state with it. Run it alongside
//! [`crate::LoudnessMeter`] on the same audio buffer to catch *sustained*
//! digital overs, which complements (rather than duplicates) the 4×-oversampled
//! true-peak detector's inter-sample peak measurement. See also
//! [`crate::rms_envelope`] and [`crate::silence_detect`] for the other two
//! complementary, independently-run QC modules.

/// Threshold mode for detecting clip events.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ClipThreshold {
    /// Digital full scale (absolute value >= 1.0).
    DigitalFullScale,
    /// Custom threshold in linear scale.
    Linear(f64),
    /// Custom threshold in dBFS.
    Dbfs(f64),
}

impl ClipThreshold {
    /// Convert this threshold to linear scale.
    #[allow(clippy::cast_precision_loss)]
    pub fn to_linear(&self) -> f64 {
        match self {
            Self::DigitalFullScale => 1.0,
            Self::Linear(v) => *v,
            Self::Dbfs(db) => 10.0_f64.powf(*db / 20.0),
        }
    }
}

impl Default for ClipThreshold {
    fn default() -> Self {
        Self::DigitalFullScale
    }
}

/// A single clip event record.
#[derive(Clone, Debug, PartialEq)]
pub struct ClipEvent {
    /// Channel index where the clip occurred.
    pub channel: usize,
    /// Sample offset (from the start of processing) where the clip began.
    pub start_sample: u64,
    /// Number of consecutive clipped samples.
    pub duration_samples: u64,
    /// Peak value observed during this clip event.
    pub peak_value: f64,
}

impl ClipEvent {
    /// Duration of the clip event in seconds.
    #[allow(clippy::cast_precision_loss)]
    pub fn duration_seconds(&self, sample_rate: f64) -> f64 {
        self.duration_samples as f64 / sample_rate
    }
}

/// Per-channel clip state tracking.
#[derive(Clone, Debug)]
struct ChannelClipState {
    /// Whether we are currently inside a clip event.
    in_clip: bool,
    /// Start sample of the current clip event.
    clip_start: u64,
    /// Duration of the current clip event.
    clip_duration: u64,
    /// Peak value of the current clip event.
    clip_peak: f64,
    /// Total number of clipped samples.
    total_clipped_samples: u64,
    /// Number of clip events.
    clip_event_count: u64,
    /// Total samples processed on this channel.
    samples_processed: u64,
}

impl ChannelClipState {
    /// Create a new channel state.
    fn new() -> Self {
        Self {
            in_clip: false,
            clip_start: 0,
            clip_duration: 0,
            clip_peak: 0.0,
            total_clipped_samples: 0,
            clip_event_count: 0,
            samples_processed: 0,
        }
    }

    /// Reset to initial state.
    fn reset(&mut self) {
        *self = Self::new();
    }
}

/// Audio clip counter and event tracker.
///
/// Monitors one or more audio channels for clipping events, records statistics,
/// and optionally logs individual clip events for later analysis.
#[derive(Clone, Debug)]
pub struct ClipCounter {
    /// Number of channels.
    channels: usize,
    /// Sample rate in Hz.
    sample_rate: f64,
    /// Clip detection threshold (linear).
    threshold_linear: f64,
    /// Per-channel state.
    channel_states: Vec<ChannelClipState>,
    /// Logged clip events (if event logging is enabled).
    events: Vec<ClipEvent>,
    /// Whether to log individual events.
    log_events: bool,
    /// Maximum number of events to store.
    max_events: usize,
}

impl ClipCounter {
    /// Create a new clip counter.
    ///
    /// # Arguments
    ///
    /// * `channels` - Number of audio channels
    /// * `sample_rate` - Sample rate in Hz
    /// * `threshold` - Clip detection threshold
    /// * `log_events` - Whether to record individual clip events
    pub fn new(
        channels: usize,
        sample_rate: f64,
        threshold: ClipThreshold,
        log_events: bool,
    ) -> Self {
        let channels = channels.max(1);
        Self {
            channels,
            sample_rate,
            threshold_linear: threshold.to_linear(),
            channel_states: (0..channels).map(|_| ChannelClipState::new()).collect(),
            events: Vec::new(),
            log_events,
            max_events: 10_000,
        }
    }

    /// Process interleaved audio samples.
    ///
    /// Samples are expected to be interleaved: [ch0, ch1, ..., ch0, ch1, ...].
    pub fn process_interleaved(&mut self, samples: &[f64]) {
        let frame_count = samples.len() / self.channels;
        for frame in 0..frame_count {
            for ch in 0..self.channels {
                let sample = samples[frame * self.channels + ch].abs();
                let state = &mut self.channel_states[ch];
                let clipping = sample >= self.threshold_linear;

                if clipping {
                    state.total_clipped_samples += 1;
                    if !state.in_clip {
                        // Start new clip event
                        state.in_clip = true;
                        state.clip_start = state.samples_processed;
                        state.clip_duration = 1;
                        state.clip_peak = sample;
                        state.clip_event_count += 1;
                    } else {
                        state.clip_duration += 1;
                        if sample > state.clip_peak {
                            state.clip_peak = sample;
                        }
                    }
                } else if state.in_clip {
                    // End clip event
                    if self.log_events && self.events.len() < self.max_events {
                        self.events.push(ClipEvent {
                            channel: ch,
                            start_sample: state.clip_start,
                            duration_samples: state.clip_duration,
                            peak_value: state.clip_peak,
                        });
                    }
                    state.in_clip = false;
                }
                state.samples_processed += 1;
            }
        }
    }

    /// Process a single channel of samples.
    pub fn process_channel(&mut self, channel: usize, samples: &[f64]) {
        if channel >= self.channels {
            return;
        }
        for &sample in samples {
            let abs_val = sample.abs();
            let state = &mut self.channel_states[channel];
            let clipping = abs_val >= self.threshold_linear;

            if clipping {
                state.total_clipped_samples += 1;
                if !state.in_clip {
                    state.in_clip = true;
                    state.clip_start = state.samples_processed;
                    state.clip_duration = 1;
                    state.clip_peak = abs_val;
                    state.clip_event_count += 1;
                } else {
                    state.clip_duration += 1;
                    if abs_val > state.clip_peak {
                        state.clip_peak = abs_val;
                    }
                }
            } else if state.in_clip {
                if self.log_events && self.events.len() < self.max_events {
                    self.events.push(ClipEvent {
                        channel,
                        start_sample: state.clip_start,
                        duration_samples: state.clip_duration,
                        peak_value: state.clip_peak,
                    });
                }
                state.in_clip = false;
            }
            state.samples_processed += 1;
        }
    }

    /// Total number of clip events across all channels.
    pub fn total_clip_events(&self) -> u64 {
        self.channel_states.iter().map(|s| s.clip_event_count).sum()
    }

    /// Number of clip events for a specific channel.
    pub fn channel_clip_events(&self, channel: usize) -> u64 {
        self.channel_states
            .get(channel)
            .map_or(0, |s| s.clip_event_count)
    }

    /// Total number of clipped samples across all channels.
    pub fn total_clipped_samples(&self) -> u64 {
        self.channel_states
            .iter()
            .map(|s| s.total_clipped_samples)
            .sum()
    }

    /// Percentage of samples that were clipped (across all channels).
    #[allow(clippy::cast_precision_loss)]
    pub fn clip_percentage(&self) -> f64 {
        let total_samples: u64 = self
            .channel_states
            .iter()
            .map(|s| s.samples_processed)
            .sum();
        if total_samples == 0 {
            return 0.0;
        }
        let clipped: u64 = self.total_clipped_samples();
        clipped as f64 / total_samples as f64 * 100.0
    }

    /// Whether any clipping has been detected.
    pub fn has_clipping(&self) -> bool {
        self.total_clip_events() > 0
    }

    /// Get logged clip events.
    pub fn events(&self) -> &[ClipEvent] {
        &self.events
    }

    /// Get the clip severity rating.
    pub fn severity(&self) -> ClipSeverity {
        let pct = self.clip_percentage();
        if pct == 0.0 {
            ClipSeverity::None
        } else if pct < 0.01 {
            ClipSeverity::Minor
        } else if pct < 0.1 {
            ClipSeverity::Moderate
        } else {
            ClipSeverity::Severe
        }
    }

    /// Reset all counters and events.
    pub fn reset(&mut self) {
        for state in &mut self.channel_states {
            state.reset();
        }
        self.events.clear();
    }
}

/// Severity classification for clipping.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClipSeverity {
    /// No clipping detected.
    None,
    /// Very occasional clips (< 0.01%).
    Minor,
    /// Noticeable clipping (< 0.1%).
    Moderate,
    /// Heavy clipping (>= 0.1%).
    Severe,
}

impl ClipSeverity {
    /// Human-readable description.
    pub fn description(&self) -> &str {
        match self {
            Self::None => "No clipping",
            Self::Minor => "Minor clipping (barely audible)",
            Self::Moderate => "Moderate clipping (audible distortion)",
            Self::Severe => "Severe clipping (significant distortion)",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_threshold_digital_full_scale() {
        let t = ClipThreshold::DigitalFullScale;
        assert!((t.to_linear() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_threshold_dbfs() {
        let t = ClipThreshold::Dbfs(-6.0);
        let linear = t.to_linear();
        // -6 dBFS ~ 0.5012
        assert!((linear - 0.5012).abs() < 0.01);
    }

    #[test]
    fn test_threshold_linear() {
        let t = ClipThreshold::Linear(0.95);
        assert!((t.to_linear() - 0.95).abs() < 1e-12);
    }

    #[test]
    fn test_no_clipping() {
        let mut counter = ClipCounter::new(1, 48000.0, ClipThreshold::DigitalFullScale, false);
        let samples: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.001).sin() * 0.5).collect();
        counter.process_channel(0, &samples);
        assert_eq!(counter.total_clip_events(), 0);
        assert!(!counter.has_clipping());
        assert_eq!(counter.severity(), ClipSeverity::None);
    }

    #[test]
    fn test_single_clip_event() {
        let mut counter = ClipCounter::new(1, 48000.0, ClipThreshold::DigitalFullScale, true);
        let mut samples = vec![0.5; 100];
        // Insert 5 clipped samples
        for s in samples.iter_mut().take(55).skip(50) {
            *s = 1.0;
        }
        // Add non-clipped tail to close the event
        samples.push(0.1);
        counter.process_channel(0, &samples);
        assert_eq!(counter.total_clip_events(), 1);
        assert_eq!(counter.events().len(), 1);
        assert_eq!(counter.events()[0].duration_samples, 5);
    }

    #[test]
    fn test_multiple_clip_events() {
        let mut counter = ClipCounter::new(1, 48000.0, ClipThreshold::DigitalFullScale, true);
        // Two separate clip bursts
        let mut samples = vec![0.5; 200];
        samples[10] = 1.0;
        samples[11] = 1.0;
        // gap
        samples[100] = 1.0;
        samples[101] = 1.0;
        samples[102] = 1.0;
        counter.process_channel(0, &samples);
        assert_eq!(counter.total_clip_events(), 2);
        assert_eq!(counter.events().len(), 2);
    }

    #[test]
    fn test_interleaved_stereo() {
        let mut counter = ClipCounter::new(2, 48000.0, ClipThreshold::DigitalFullScale, true);
        // Interleaved: [L, R, L, R, ...]
        let mut samples = vec![0.5; 20]; // 10 frames
                                         // Clip on left channel at frame 3
        samples[6] = 1.0; // frame 3, ch 0
                          // Clip on right channel at frame 5
        samples[11] = 1.0; // frame 5, ch 1
                           // Non-clipped tail
        samples.extend_from_slice(&[0.1, 0.1]);
        counter.process_interleaved(&samples);
        assert_eq!(counter.channel_clip_events(0), 1);
        assert_eq!(counter.channel_clip_events(1), 1);
        assert_eq!(counter.total_clip_events(), 2);
    }

    #[test]
    fn test_clip_percentage() {
        let mut counter = ClipCounter::new(1, 48000.0, ClipThreshold::DigitalFullScale, false);
        let mut samples = vec![0.5; 1000];
        // 10 clipped samples out of 1000 = 1%
        for s in samples.iter_mut().take(10) {
            *s = 1.0;
        }
        counter.process_channel(0, &samples);
        let pct = counter.clip_percentage();
        assert!((pct - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_severity_levels() {
        assert_eq!(ClipSeverity::None.description(), "No clipping");
        assert_eq!(
            ClipSeverity::Severe.description(),
            "Severe clipping (significant distortion)"
        );
    }

    #[test]
    fn test_clip_event_duration_seconds() {
        let evt = ClipEvent {
            channel: 0,
            start_sample: 0,
            duration_samples: 48000,
            peak_value: 1.0,
        };
        assert!((evt.duration_seconds(48000.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_reset() {
        let mut counter = ClipCounter::new(1, 48000.0, ClipThreshold::DigitalFullScale, true);
        counter.process_channel(0, &[1.0, 1.0, 0.5]);
        assert!(counter.has_clipping());
        counter.reset();
        assert!(!counter.has_clipping());
        assert!(counter.events().is_empty());
    }

    #[test]
    fn test_custom_threshold() {
        let mut counter = ClipCounter::new(1, 48000.0, ClipThreshold::Linear(0.8), true);
        counter.process_channel(0, &[0.9, 0.85, 0.5]);
        assert_eq!(counter.total_clip_events(), 1);
        assert_eq!(counter.events()[0].duration_samples, 2);
    }

    #[test]
    fn test_invalid_channel_ignored() {
        let mut counter = ClipCounter::new(2, 48000.0, ClipThreshold::DigitalFullScale, false);
        // Channel 5 does not exist; should be silently ignored
        counter.process_channel(5, &[1.0, 1.0]);
        assert_eq!(counter.total_clip_events(), 0);
    }
}
