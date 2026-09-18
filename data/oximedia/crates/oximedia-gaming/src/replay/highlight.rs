//! Replay highlight detection and event recording.
//!
//! Records gameplay events for replay analysis, provides playback control,
//! and automatically detects highlights based on configurable rules.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::VecDeque;
use std::time::Duration;

/// A single gameplay event recorded for replay analysis.
#[derive(Debug, Clone)]
pub struct GameEvent {
    /// Event identifier
    pub id: u64,
    /// Timestamp offset from session start
    pub timestamp: Duration,
    /// Category of the event
    pub category: EventCategory,
    /// Human-readable description
    pub description: String,
    /// Significance score (0.0 = low, 1.0 = maximum)
    pub significance: f32,
}

impl GameEvent {
    /// Create a new game event.
    #[must_use]
    pub fn new(
        id: u64,
        timestamp: Duration,
        category: EventCategory,
        description: impl Into<String>,
        significance: f32,
    ) -> Self {
        Self {
            id,
            timestamp,
            category,
            description: description.into(),
            significance: significance.clamp(0.0, 1.0),
        }
    }

    /// Returns true if this event is considered a highlight.
    #[must_use]
    pub fn is_highlight(&self, threshold: f32) -> bool {
        self.significance >= threshold
    }
}

/// Categories of gameplay events.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EventCategory {
    /// Combat kill or elimination
    Kill,
    /// Player death
    Death,
    /// Objective captured
    ObjectiveCaptured,
    /// Match started
    MatchStart,
    /// Match ended
    MatchEnd,
    /// Score milestone reached
    ScoreMilestone,
    /// Custom event
    Custom(String),
}

impl EventCategory {
    /// Returns the default significance for events of this category.
    #[must_use]
    pub fn default_significance(&self) -> f32 {
        match self {
            Self::Kill => 0.7,
            Self::Death => 0.3,
            Self::ObjectiveCaptured => 0.9,
            Self::MatchStart => 0.5,
            Self::MatchEnd => 0.6,
            Self::ScoreMilestone => 0.8,
            Self::Custom(_) => 0.5,
        }
    }
}

/// Playback state of the replay controller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaybackState {
    /// Not playing
    Stopped,
    /// Playing at normal speed
    Playing,
    /// Paused
    Paused,
    /// Fast forward
    FastForward,
    /// Rewind
    Rewind,
}

/// Controls replay playback with speed adjustment and seeking.
#[derive(Debug)]
pub struct PlaybackController {
    state: PlaybackState,
    /// Current position in the recording
    position: Duration,
    /// Total recording duration
    total_duration: Duration,
    /// Playback speed multiplier (1.0 = normal)
    speed: f32,
}

impl PlaybackController {
    /// Create a new playback controller.
    #[must_use]
    pub fn new(total_duration: Duration) -> Self {
        Self {
            state: PlaybackState::Stopped,
            position: Duration::ZERO,
            total_duration,
            speed: 1.0,
        }
    }

    /// Start playing.
    pub fn play(&mut self) {
        self.state = PlaybackState::Playing;
        self.speed = 1.0;
    }

    /// Pause playback.
    pub fn pause(&mut self) {
        if self.state == PlaybackState::Playing {
            self.state = PlaybackState::Paused;
        }
    }

    /// Stop and reset to beginning.
    pub fn stop(&mut self) {
        self.state = PlaybackState::Stopped;
        self.position = Duration::ZERO;
    }

    /// Set fast-forward speed (must be > 1.0).
    pub fn fast_forward(&mut self, speed: f32) {
        self.state = PlaybackState::FastForward;
        self.speed = speed.max(1.1);
    }

    /// Set rewind speed (must be > 1.0).
    pub fn rewind(&mut self, speed: f32) {
        self.state = PlaybackState::Rewind;
        self.speed = speed.max(1.1);
    }

    /// Seek to a specific position (clamped to valid range).
    pub fn seek(&mut self, pos: Duration) {
        self.position = pos.min(self.total_duration);
    }

    /// Advance the controller by the given elapsed real time.
    pub fn advance(&mut self, elapsed: Duration) {
        match self.state {
            PlaybackState::Playing | PlaybackState::FastForward => {
                let delta = Duration::from_secs_f64(elapsed.as_secs_f64() * f64::from(self.speed));
                self.position = (self.position + delta).min(self.total_duration);
                if self.position == self.total_duration {
                    self.state = PlaybackState::Stopped;
                }
            }
            PlaybackState::Rewind => {
                let delta = Duration::from_secs_f64(elapsed.as_secs_f64() * f64::from(self.speed));
                self.position = self.position.saturating_sub(delta);
                if self.position.is_zero() {
                    self.state = PlaybackState::Stopped;
                }
            }
            PlaybackState::Paused | PlaybackState::Stopped => {}
        }
    }

    /// Current position in the replay.
    #[must_use]
    pub fn position(&self) -> Duration {
        self.position
    }

    /// Current playback state.
    #[must_use]
    pub fn state(&self) -> PlaybackState {
        self.state
    }

    /// Current playback speed.
    #[must_use]
    pub fn speed(&self) -> f32 {
        self.speed
    }

    /// Progress as a fraction 0.0–1.0.
    #[must_use]
    pub fn progress(&self) -> f32 {
        if self.total_duration.is_zero() {
            return 0.0;
        }
        (self.position.as_secs_f64() / self.total_duration.as_secs_f64()) as f32
    }
}

/// Configuration for the highlight detector.
#[derive(Debug, Clone)]
pub struct HighlightConfig {
    /// Minimum significance score to qualify as a highlight
    pub significance_threshold: f32,
    /// Window before the event to include in the highlight clip
    pub pre_event_window: Duration,
    /// Window after the event to include in the highlight clip
    pub post_event_window: Duration,
    /// Minimum gap between consecutive highlights
    pub min_gap: Duration,
}

impl Default for HighlightConfig {
    fn default() -> Self {
        Self {
            significance_threshold: 0.6,
            pre_event_window: Duration::from_secs(5),
            post_event_window: Duration::from_secs(10),
            min_gap: Duration::from_secs(15),
        }
    }
}

/// A detected highlight clip with start/end times.
#[derive(Debug, Clone)]
pub struct HighlightClip {
    /// Triggering event
    pub trigger_event: GameEvent,
    /// Clip start time
    pub start: Duration,
    /// Clip end time
    pub end: Duration,
}

impl HighlightClip {
    /// Duration of this highlight clip.
    #[must_use]
    pub fn duration(&self) -> Duration {
        self.end.saturating_sub(self.start)
    }
}

/// Detects highlight moments from a sequence of game events.
#[derive(Debug, Default)]
pub struct HighlightDetector {
    config: HighlightConfig,
    events: VecDeque<GameEvent>,
    next_id: u64,
}

impl HighlightDetector {
    /// Create a new detector with default config.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a detector with custom configuration.
    #[must_use]
    pub fn with_config(config: HighlightConfig) -> Self {
        Self {
            config,
            events: VecDeque::new(),
            next_id: 1,
        }
    }

    /// Record a new event and return its id.
    pub fn record_event(
        &mut self,
        timestamp: Duration,
        category: EventCategory,
        description: impl Into<String>,
    ) -> u64 {
        let sig = category.default_significance();
        self.next_id += 1;
        let id = self.next_id;
        let event = GameEvent::new(id, timestamp, category, description, sig);
        self.events.push_back(event);
        id
    }

    /// Record an event with an explicit significance score.
    pub fn record_event_with_significance(
        &mut self,
        timestamp: Duration,
        category: EventCategory,
        description: impl Into<String>,
        significance: f32,
    ) -> u64 {
        self.next_id += 1;
        let id = self.next_id;
        let event = GameEvent::new(id, timestamp, category, description, significance);
        self.events.push_back(event);
        id
    }

    /// Run highlight detection and return discovered clips.
    #[must_use]
    pub fn detect_highlights(&self) -> Vec<HighlightClip> {
        let mut clips: Vec<HighlightClip> = Vec::new();
        let threshold = self.config.significance_threshold;
        let mut last_highlight_end: Option<Duration> = None;

        let mut sorted: Vec<&GameEvent> = self.events.iter().collect();
        sorted.sort_by_key(|e| e.timestamp);

        for event in sorted {
            if !event.is_highlight(threshold) {
                continue;
            }
            let clip_start = event.timestamp.saturating_sub(self.config.pre_event_window);
            let clip_end = event.timestamp + self.config.post_event_window;

            // Enforce minimum gap between highlights
            if let Some(prev_end) = last_highlight_end {
                if clip_start < prev_end + self.config.min_gap {
                    continue;
                }
            }

            last_highlight_end = Some(clip_end);
            clips.push(HighlightClip {
                trigger_event: event.clone(),
                start: clip_start,
                end: clip_end,
            });
        }
        clips
    }

    /// Total events recorded.
    #[must_use]
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Get all events with significance above the configured threshold.
    #[must_use]
    pub fn significant_events(&self) -> Vec<&GameEvent> {
        self.events
            .iter()
            .filter(|e| e.is_highlight(self.config.significance_threshold))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_game_event_is_highlight() {
        let event = GameEvent::new(1, Duration::from_secs(10), EventCategory::Kill, "kill", 0.7);
        assert!(event.is_highlight(0.6));
        assert!(!event.is_highlight(0.8));
    }

    #[test]
    fn test_game_event_significance_clamp() {
        let event = GameEvent::new(1, Duration::from_secs(0), EventCategory::Kill, "k", 2.0);
        assert_eq!(event.significance, 1.0);
    }

    #[test]
    fn test_event_category_default_significance() {
        assert!(EventCategory::ObjectiveCaptured.default_significance() >= 0.8);
        assert!(EventCategory::Death.default_significance() < 0.5);
    }

    #[test]
    fn test_playback_controller_play_and_advance() {
        let mut ctrl = PlaybackController::new(Duration::from_mins(1));
        ctrl.play();
        assert_eq!(ctrl.state(), PlaybackState::Playing);
        ctrl.advance(Duration::from_secs(10));
        assert_eq!(ctrl.position(), Duration::from_secs(10));
    }

    #[test]
    fn test_playback_controller_pause() {
        let mut ctrl = PlaybackController::new(Duration::from_mins(1));
        ctrl.play();
        ctrl.pause();
        ctrl.advance(Duration::from_secs(5));
        assert_eq!(ctrl.position(), Duration::ZERO);
    }

    #[test]
    fn test_playback_controller_stop_resets() {
        let mut ctrl = PlaybackController::new(Duration::from_mins(1));
        ctrl.play();
        ctrl.advance(Duration::from_secs(20));
        ctrl.stop();
        assert_eq!(ctrl.position(), Duration::ZERO);
        assert_eq!(ctrl.state(), PlaybackState::Stopped);
    }

    #[test]
    fn test_playback_controller_seek() {
        let mut ctrl = PlaybackController::new(Duration::from_mins(1));
        ctrl.seek(Duration::from_secs(30));
        assert_eq!(ctrl.position(), Duration::from_secs(30));
    }

    #[test]
    fn test_playback_controller_seek_clamps_to_end() {
        let mut ctrl = PlaybackController::new(Duration::from_mins(1));
        ctrl.seek(Duration::from_secs(100));
        assert_eq!(ctrl.position(), Duration::from_mins(1));
    }

    #[test]
    fn test_playback_controller_fast_forward() {
        let mut ctrl = PlaybackController::new(Duration::from_mins(1));
        ctrl.fast_forward(2.0);
        ctrl.advance(Duration::from_secs(5));
        // 5s * 2x speed = 10s
        assert_eq!(ctrl.position(), Duration::from_secs(10));
    }

    #[test]
    fn test_playback_controller_rewind() {
        let mut ctrl = PlaybackController::new(Duration::from_mins(1));
        ctrl.seek(Duration::from_secs(30));
        ctrl.rewind(2.0);
        ctrl.advance(Duration::from_secs(5));
        // 30 - (5 * 2.0) = 20
        assert_eq!(ctrl.position(), Duration::from_secs(20));
    }

    #[test]
    fn test_playback_controller_progress() {
        let mut ctrl = PlaybackController::new(Duration::from_secs(100));
        ctrl.seek(Duration::from_secs(50));
        assert!((ctrl.progress() - 0.5).abs() < 1e-3);
    }

    #[test]
    fn test_highlight_detector_record_events() {
        let mut detector = HighlightDetector::new();
        let id1 = detector.record_event(Duration::from_secs(10), EventCategory::Kill, "kill");
        let id2 = detector.record_event(Duration::from_secs(20), EventCategory::Death, "death");
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(detector.event_count(), 2);
    }

    #[test]
    fn test_highlight_detection_basic() {
        let mut detector = HighlightDetector::with_config(HighlightConfig {
            significance_threshold: 0.6,
            pre_event_window: Duration::from_secs(5),
            post_event_window: Duration::from_secs(10),
            min_gap: Duration::from_secs(30),
        });
        // Kill (0.7) should trigger a highlight
        detector.record_event(Duration::from_mins(1), EventCategory::Kill, "epic kill");
        // Death (0.3) should not trigger a highlight
        detector.record_event(Duration::from_mins(2), EventCategory::Death, "death");
        let clips = detector.detect_highlights();
        assert_eq!(clips.len(), 1);
        assert_eq!(clips[0].trigger_event.category, EventCategory::Kill);
    }

    #[test]
    fn test_highlight_clip_duration() {
        let config = HighlightConfig {
            significance_threshold: 0.5,
            pre_event_window: Duration::from_secs(5),
            post_event_window: Duration::from_secs(10),
            min_gap: Duration::from_secs(30),
        };
        let mut detector = HighlightDetector::with_config(config);
        detector.record_event(
            Duration::from_secs(100),
            EventCategory::ObjectiveCaptured,
            "obj",
        );
        let clips = detector.detect_highlights();
        assert_eq!(clips.len(), 1);
        assert_eq!(clips[0].duration(), Duration::from_secs(15));
    }

    #[test]
    fn test_highlight_min_gap_filter() {
        let config = HighlightConfig {
            significance_threshold: 0.5,
            pre_event_window: Duration::from_secs(3),
            post_event_window: Duration::from_secs(5),
            min_gap: Duration::from_secs(30),
        };
        let mut detector = HighlightDetector::with_config(config);
        // Two kills close together - only first should become a highlight
        detector.record_event(Duration::from_secs(10), EventCategory::Kill, "kill1");
        detector.record_event(Duration::from_secs(15), EventCategory::Kill, "kill2");
        let clips = detector.detect_highlights();
        assert_eq!(clips.len(), 1);
    }

    #[test]
    fn test_significant_events_filter() {
        let mut detector = HighlightDetector::new();
        detector.record_event(Duration::from_secs(10), EventCategory::Kill, "k");
        detector.record_event(Duration::from_secs(20), EventCategory::Death, "d");
        detector.record_event(
            Duration::from_secs(30),
            EventCategory::ObjectiveCaptured,
            "o",
        );
        let significant = detector.significant_events();
        // Kill (0.7) and Objective (0.9) exceed default threshold of 0.6; Death (0.3) does not
        assert!(significant
            .iter()
            .any(|e| e.category == EventCategory::Kill));
        assert!(significant
            .iter()
            .any(|e| e.category == EventCategory::ObjectiveCaptured));
        assert!(!significant
            .iter()
            .any(|e| e.category == EventCategory::Death));
    }
}
