//! Audio event detection (silence, speech, music, applause transitions).
//!
//! Provides a streaming `AudioEventDetector` that classifies short-time
//! audio frames into event types and emits `AudioEvent` records on transitions.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// Classification of audio event types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AudioEventType {
    /// Audio level has dropped below the silence threshold.
    SilenceStart,
    /// Audio level has risen above silence threshold after a silent period.
    SilenceEnd,
    /// Speech activity has been detected.
    SpeechStart,
    /// Music content has been detected.
    MusicStart,
    /// Applause / crowd noise has been detected.
    ApplauseStart,
    /// Generic loud transient (e.g. clap, gunshot).
    LoudTransient,
}

impl AudioEventType {
    /// Short human-readable label for this event type.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::SilenceStart => "silence_start",
            Self::SilenceEnd => "silence_end",
            Self::SpeechStart => "speech_start",
            Self::MusicStart => "music_start",
            Self::ApplauseStart => "applause_start",
            Self::LoudTransient => "loud_transient",
        }
    }

    /// Returns `true` for events that mark the beginning of active content.
    #[must_use]
    pub fn is_activity_start(&self) -> bool {
        matches!(
            self,
            Self::SilenceEnd
                | Self::SpeechStart
                | Self::MusicStart
                | Self::ApplauseStart
                | Self::LoudTransient
        )
    }
}

/// A detected audio event with timing information.
#[derive(Debug, Clone)]
pub struct AudioEvent {
    /// Type of event.
    pub event_type: AudioEventType,
    /// Time at which the event was detected (seconds).
    pub time_s: f32,
    /// Duration of the event segment in milliseconds (may be 0 for instantaneous events).
    pub duration_ms: f32,
    /// Confidence of the detection in [0, 1].
    pub confidence: f32,
}

impl AudioEvent {
    /// Create a new audio event.
    #[must_use]
    pub fn new(event_type: AudioEventType, time_s: f32, duration_ms: f32, confidence: f32) -> Self {
        Self {
            event_type,
            time_s,
            duration_ms,
            confidence,
        }
    }

    /// Duration in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> f32 {
        self.duration_ms
    }

    /// Duration in seconds.
    #[must_use]
    pub fn duration_s(&self) -> f32 {
        self.duration_ms / 1000.0
    }

    /// Whether this event spans a meaningful duration (> 0 ms).
    #[must_use]
    pub fn has_duration(&self) -> bool {
        self.duration_ms > 0.0
    }
}

/// Per-frame feature summary fed into the event detector.
#[derive(Debug, Clone, Copy)]
pub struct AudioFrame {
    /// RMS energy of the frame (linear).
    pub rms: f32,
    /// Zero-crossing rate normalised to [0, 1].
    pub zcr: f32,
    /// Spectral centroid normalised to [0, 1] (relative to Nyquist).
    pub centroid_norm: f32,
    /// Spectral flux relative to previous frame.
    pub flux: f32,
    /// Frame timestamp in seconds.
    pub time_s: f32,
}

impl AudioFrame {
    /// Create a new audio frame summary.
    #[must_use]
    pub fn new(rms: f32, zcr: f32, centroid_norm: f32, flux: f32, time_s: f32) -> Self {
        Self {
            rms,
            zcr,
            centroid_norm,
            flux,
            time_s,
        }
    }
}

/// Configuration thresholds for the event detector.
#[derive(Debug, Clone)]
pub struct EventDetectorConfig {
    /// RMS below which audio is considered silent.
    pub silence_threshold: f32,
    /// Minimum number of consecutive silent frames before `SilenceStart` is emitted.
    pub silence_min_frames: usize,
    /// ZCR above which content is classified as speech.
    pub speech_zcr_threshold: f32,
    /// Spectral centroid below which content is classified as music.
    pub music_centroid_max: f32,
    /// Flux threshold for loud-transient detection.
    pub transient_flux_threshold: f32,
}

impl Default for EventDetectorConfig {
    fn default() -> Self {
        Self {
            silence_threshold: 0.01,
            silence_min_frames: 10,
            speech_zcr_threshold: 0.15,
            music_centroid_max: 0.4,
            transient_flux_threshold: 5.0,
        }
    }
}

/// Streaming audio event detector.
///
/// Call `add_frame()` for each analysis frame; query `events()` to retrieve
/// all transitions detected so far.
#[derive(Debug, Clone)]
pub struct AudioEventDetector {
    config: EventDetectorConfig,
    events: Vec<AudioEvent>,
    /// Count of consecutive silent frames seen.
    silent_frame_count: usize,
    /// Whether we are currently in a silent state.
    in_silence: bool,
    frame_count: usize,
}

impl AudioEventDetector {
    /// Create a new detector with the given configuration.
    #[must_use]
    pub fn new(config: EventDetectorConfig) -> Self {
        Self {
            config,
            events: Vec::new(),
            silent_frame_count: 0,
            in_silence: false,
            frame_count: 0,
        }
    }

    /// Create a new detector with default thresholds.
    #[must_use]
    pub fn default_detector() -> Self {
        Self::new(EventDetectorConfig::default())
    }

    /// Feed one analysis frame to the detector.
    ///
    /// Internally classifies the frame and emits events on state transitions.
    pub fn add_frame(&mut self, frame: AudioFrame) {
        self.frame_count += 1;

        // --- Silence detection ---
        if frame.rms < self.config.silence_threshold {
            self.silent_frame_count += 1;
            if self.silent_frame_count == self.config.silence_min_frames && !self.in_silence {
                self.in_silence = true;
                self.events.push(AudioEvent::new(
                    AudioEventType::SilenceStart,
                    frame.time_s,
                    0.0,
                    0.95,
                ));
            }
            return; // Don't classify silent frames as speech/music
        }

        // Frame is non-silent
        if self.in_silence {
            // Transition out of silence
            self.in_silence = false;
            self.events.push(AudioEvent::new(
                AudioEventType::SilenceEnd,
                frame.time_s,
                0.0,
                0.95,
            ));
        }
        self.silent_frame_count = 0;

        // --- Loud transient ---
        if frame.flux > self.config.transient_flux_threshold {
            self.events.push(AudioEvent::new(
                AudioEventType::LoudTransient,
                frame.time_s,
                10.0, // instantaneous, ~10 ms
                (frame.flux / (self.config.transient_flux_threshold * 2.0)).min(1.0),
            ));
            return;
        }

        // --- Applause heuristic: moderate ZCR + high flux ---
        if frame.zcr > 0.25 && frame.flux > 1.5 {
            self.events.push(AudioEvent::new(
                AudioEventType::ApplauseStart,
                frame.time_s,
                0.0,
                0.6,
            ));
            return;
        }

        // --- Speech vs music classification ---
        if frame.zcr >= self.config.speech_zcr_threshold {
            self.events.push(AudioEvent::new(
                AudioEventType::SpeechStart,
                frame.time_s,
                0.0,
                (frame.zcr * 2.0).min(1.0),
            ));
        } else if frame.centroid_norm <= self.config.music_centroid_max {
            self.events.push(AudioEvent::new(
                AudioEventType::MusicStart,
                frame.time_s,
                0.0,
                1.0 - frame.centroid_norm / self.config.music_centroid_max,
            ));
        }
    }

    /// Return all events detected so far.
    #[must_use]
    pub fn events(&self) -> &[AudioEvent] {
        &self.events
    }

    /// Return the total number of frames processed.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.frame_count
    }

    /// Return events of a specific type.
    #[must_use]
    pub fn events_of_type(&self, event_type: AudioEventType) -> Vec<&AudioEvent> {
        self.events
            .iter()
            .filter(|e| e.event_type == event_type)
            .collect()
    }

    /// Clear all accumulated events.
    pub fn reset_events(&mut self) {
        self.events.clear();
        self.silent_frame_count = 0;
        self.in_silence = false;
        self.frame_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn silent_frame(t: f32) -> AudioFrame {
        AudioFrame::new(0.0, 0.0, 0.0, 0.0, t)
    }

    fn speech_frame(t: f32) -> AudioFrame {
        AudioFrame::new(0.1, 0.2, 0.5, 0.5, t)
    }

    fn music_frame(t: f32) -> AudioFrame {
        AudioFrame::new(0.15, 0.05, 0.3, 0.3, t)
    }

    fn loud_frame(t: f32) -> AudioFrame {
        AudioFrame::new(0.9, 0.3, 0.5, 10.0, t)
    }

    // AudioEventType tests

    #[test]
    fn test_labels_non_empty() {
        let types = [
            AudioEventType::SilenceStart,
            AudioEventType::SilenceEnd,
            AudioEventType::SpeechStart,
            AudioEventType::MusicStart,
            AudioEventType::ApplauseStart,
            AudioEventType::LoudTransient,
        ];
        for t in types {
            assert!(!t.label().is_empty());
        }
    }

    #[test]
    fn test_silence_start_not_activity() {
        assert!(!AudioEventType::SilenceStart.is_activity_start());
    }

    #[test]
    fn test_silence_end_is_activity() {
        assert!(AudioEventType::SilenceEnd.is_activity_start());
    }

    #[test]
    fn test_speech_start_is_activity() {
        assert!(AudioEventType::SpeechStart.is_activity_start());
    }

    // AudioEvent tests

    #[test]
    fn test_event_duration_ms() {
        let ev = AudioEvent::new(AudioEventType::SpeechStart, 1.0, 250.0, 0.8);
        assert!((ev.duration_ms() - 250.0).abs() < 1e-5);
    }

    #[test]
    fn test_event_duration_s() {
        let ev = AudioEvent::new(AudioEventType::MusicStart, 2.0, 500.0, 0.9);
        assert!((ev.duration_s() - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_event_has_duration_false_when_zero() {
        let ev = AudioEvent::new(AudioEventType::SilenceStart, 0.0, 0.0, 1.0);
        assert!(!ev.has_duration());
    }

    // AudioEventDetector tests

    #[test]
    fn test_no_events_initially() {
        let det = AudioEventDetector::default_detector();
        assert!(det.events().is_empty());
    }

    #[test]
    fn test_silence_start_emitted_after_min_frames() {
        let mut det = AudioEventDetector::default_detector();
        for i in 0..15 {
            det.add_frame(silent_frame(i as f32 * 0.023));
        }
        let silence_starts = det.events_of_type(AudioEventType::SilenceStart);
        assert!(!silence_starts.is_empty());
    }

    #[test]
    fn test_silence_end_emitted_after_active_frame() {
        let mut det = AudioEventDetector::default_detector();
        for i in 0..15 {
            det.add_frame(silent_frame(i as f32 * 0.023));
        }
        det.add_frame(speech_frame(0.5));
        let ends = det.events_of_type(AudioEventType::SilenceEnd);
        assert!(!ends.is_empty());
    }

    #[test]
    fn test_speech_frame_classified() {
        let mut det = AudioEventDetector::default_detector();
        det.add_frame(speech_frame(0.0));
        let speech = det.events_of_type(AudioEventType::SpeechStart);
        assert!(!speech.is_empty());
    }

    #[test]
    fn test_music_frame_classified() {
        let mut det = AudioEventDetector::default_detector();
        det.add_frame(music_frame(0.0));
        let music = det.events_of_type(AudioEventType::MusicStart);
        assert!(!music.is_empty());
    }

    #[test]
    fn test_loud_transient_detected() {
        let mut det = AudioEventDetector::default_detector();
        det.add_frame(loud_frame(0.0));
        let transients = det.events_of_type(AudioEventType::LoudTransient);
        assert!(!transients.is_empty());
    }

    #[test]
    fn test_frame_count_increments() {
        let mut det = AudioEventDetector::default_detector();
        for i in 0..5 {
            det.add_frame(speech_frame(i as f32 * 0.023));
        }
        assert_eq!(det.frame_count(), 5);
    }

    #[test]
    fn test_reset_clears_events() {
        let mut det = AudioEventDetector::default_detector();
        for i in 0..20 {
            det.add_frame(silent_frame(i as f32 * 0.023));
        }
        assert!(!det.events().is_empty());
        det.reset_events();
        assert!(det.events().is_empty());
        assert_eq!(det.frame_count(), 0);
    }

    #[test]
    fn test_confidence_within_bounds() {
        let mut det = AudioEventDetector::default_detector();
        for i in 0..20 {
            det.add_frame(speech_frame(i as f32 * 0.023));
        }
        for ev in det.events() {
            assert!(
                ev.confidence >= 0.0 && ev.confidence <= 1.0,
                "confidence {} out of bounds",
                ev.confidence
            );
        }
    }
}
