//! Gameplay event recording and export.
//!
//! Provides a standalone [`EventRecorder`] that accumulates [`GameEvent`]s
//! (defined here, independent of `game_event::GameEvent`) and serialises them
//! to JSON, CSV, or a compact game-specific format via [`EventFormat`].

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Taxonomy of gameplay events captured by [`EventRecorder`].
#[derive(Debug, Clone, PartialEq)]
pub enum GameEventType {
    /// A player elimination event.
    Kill,
    /// The local player was eliminated.
    Death,
    /// An item or power-up was picked up.
    Pickup,
    /// Player respawned.
    Respawn,
    /// A new level began.
    LevelStart,
    /// The current level ended.
    LevelEnd,
    /// An in-game achievement was unlocked.
    Achievement,
    /// A custom, game-specific event.
    Custom(String),
}

impl GameEventType {
    /// Human-readable name for this event type.
    #[must_use]
    pub fn name(&self) -> String {
        match self {
            Self::Kill => "Kill".to_string(),
            Self::Death => "Death".to_string(),
            Self::Pickup => "Pickup".to_string(),
            Self::Respawn => "Respawn".to_string(),
            Self::LevelStart => "LevelStart".to_string(),
            Self::LevelEnd => "LevelEnd".to_string(),
            Self::Achievement => "Achievement".to_string(),
            Self::Custom(s) => s.clone(),
        }
    }

    /// Returns `true` when both variants share the same discriminant.
    /// Two `Custom` values with different labels are still considered the
    /// same type for filtering purposes (both are `Custom`).
    #[must_use]
    pub fn same_kind(&self, other: &Self) -> bool {
        matches!(
            (self, other),
            (Self::Kill, Self::Kill)
                | (Self::Death, Self::Death)
                | (Self::Pickup, Self::Pickup)
                | (Self::Respawn, Self::Respawn)
                | (Self::LevelStart, Self::LevelStart)
                | (Self::LevelEnd, Self::LevelEnd)
                | (Self::Achievement, Self::Achievement)
                | (Self::Custom(_), Self::Custom(_))
        )
    }
}

/// A single recorded gameplay event.
#[derive(Debug, Clone)]
pub struct GameEvent {
    /// The type of event.
    pub event_type: GameEventType,
    /// Milliseconds since some epoch (e.g. session start or Unix epoch).
    pub timestamp_ms: u64,
    /// Arbitrary key/value metadata associated with the event.
    pub data: HashMap<String, String>,
}

impl GameEvent {
    /// Create a new event with no additional data.
    #[must_use]
    pub fn new(event_type: GameEventType, timestamp_ms: u64) -> Self {
        Self {
            event_type,
            timestamp_ms,
            data: HashMap::new(),
        }
    }

    /// Attach a key/value pair to the event (builder pattern).
    #[must_use]
    pub fn with_data(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.data.insert(key.into(), value.into());
        self
    }

    /// Retrieve a data value by key.
    #[must_use]
    pub fn get_data(&self, key: &str) -> Option<&String> {
        self.data.get(key)
    }
}

/// Output format for [`EventRecorder::export_events`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventFormat {
    /// JSON array of event objects.
    Json,
    /// CSV with header `type,timestamp_ms,data`.
    Csv,
    /// Compact, human-readable game-specific format.
    GameSpecific,
}

// ---------------------------------------------------------------------------
// EventRecorder
// ---------------------------------------------------------------------------

/// Append-only recorder that accumulates [`GameEvent`]s and can export them.
#[derive(Debug, Default)]
pub struct EventRecorder {
    events: Vec<GameEvent>,
}

impl EventRecorder {
    /// Create a new, empty recorder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append an event to the recorder.
    pub fn record(&mut self, event: GameEvent) {
        self.events.push(event);
    }

    /// Total number of recorded events.
    #[must_use]
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Remove all recorded events.
    pub fn clear(&mut self) {
        self.events.clear();
    }

    /// Return all events whose type shares the same kind as `event_type`.
    /// For `Custom(_)`, any `Custom` event is returned regardless of its label.
    #[must_use]
    pub fn events_of_type(&self, event_type: &GameEventType) -> Vec<&GameEvent> {
        self.events
            .iter()
            .filter(|e| e.event_type.same_kind(event_type))
            .collect()
    }

    /// Return all events with `timestamp_ms` in the closed interval
    /// `[start_ms, end_ms]`.
    #[must_use]
    pub fn events_in_range(&self, start_ms: u64, end_ms: u64) -> Vec<&GameEvent> {
        self.events
            .iter()
            .filter(|e| e.timestamp_ms >= start_ms && e.timestamp_ms <= end_ms)
            .collect()
    }

    /// Return the most recently recorded event, or `None` if empty.
    #[must_use]
    pub fn latest_event(&self) -> Option<&GameEvent> {
        self.events.last()
    }

    /// Serialise all recorded events using the requested format.
    #[must_use]
    pub fn export_events(&self, format: EventFormat) -> String {
        match format {
            EventFormat::Json => self.export_json(),
            EventFormat::Csv => self.export_csv(),
            EventFormat::GameSpecific => self.export_game_specific(),
        }
    }

    // --- private serialisers ---

    fn export_json(&self) -> String {
        let mut out = String::from('[');
        for (i, event) in self.events.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push('{');
            out.push_str(&format!(
                "\"type\":\"{}\",\"timestamp_ms\":{}",
                escape_json(&event.event_type.name()),
                event.timestamp_ms
            ));
            if !event.data.is_empty() {
                out.push_str(",\"data\":{");
                let mut pairs: Vec<_> = event.data.iter().collect();
                pairs.sort_by_key(|(k, _)| k.as_str());
                for (di, (k, v)) in pairs.iter().enumerate() {
                    if di > 0 {
                        out.push(',');
                    }
                    out.push_str(&format!("\"{}\":\"{}\"", escape_json(k), escape_json(v)));
                }
                out.push('}');
            }
            out.push('}');
        }
        out.push(']');
        out
    }

    fn export_csv(&self) -> String {
        let mut out = String::from("type,timestamp_ms,data\n");
        for event in &self.events {
            let data_str = if event.data.is_empty() {
                String::new()
            } else {
                let mut pairs: Vec<_> = event.data.iter().collect();
                pairs.sort_by_key(|(k, _)| k.as_str());
                pairs
                    .iter()
                    .map(|(k, v)| format!("{k}={v}"))
                    .collect::<Vec<_>>()
                    .join(";")
            };
            out.push_str(&format!(
                "{},{},{}\n",
                event.event_type.name(),
                event.timestamp_ms,
                data_str
            ));
        }
        out
    }

    fn export_game_specific(&self) -> String {
        let mut out = String::new();
        for event in &self.events {
            let data_str = if event.data.is_empty() {
                String::new()
            } else {
                let mut pairs: Vec<_> = event.data.iter().collect();
                pairs.sort_by_key(|(k, _)| k.as_str());
                let inner = pairs
                    .iter()
                    .map(|(k, v)| format!("{k}={v}"))
                    .collect::<Vec<_>>()
                    .join(",");
                format!(" {{{inner}}}")
            };
            out.push_str(&format!(
                "[{}] {}:{}\n",
                event.timestamp_ms,
                event.event_type.name(),
                data_str
            ));
        }
        out
    }
}

/// Escape a string for JSON output (handles `"` and `\`).
fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

// ---------------------------------------------------------------------------
// StingerTransition
// ---------------------------------------------------------------------------

/// Type of stinger transition effect.
#[derive(Debug, Clone, PartialEq)]
pub enum StingerPreset {
    /// Horizontal slide (left-to-right).
    SlideLeft,
    /// Horizontal slide (right-to-left).
    SlideRight,
    /// Vertical wipe (top-to-bottom).
    WipeDown,
    /// Vertical wipe (bottom-to-top).
    WipeUp,
    /// Cross-fade with alpha.
    Fade,
    /// Zoom in/out with blur.
    Zoom,
    /// Custom image sequence frames (paths or data).
    CustomSequence(Vec<Vec<u8>>),
}

impl StingerPreset {
    /// Human-readable name.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::SlideLeft => "SlideLeft",
            Self::SlideRight => "SlideRight",
            Self::WipeDown => "WipeDown",
            Self::WipeUp => "WipeUp",
            Self::Fade => "Fade",
            Self::Zoom => "Zoom",
            Self::CustomSequence(_) => "CustomSequence",
        }
    }

    /// Whether this preset uses custom image data.
    #[must_use]
    pub fn is_custom(&self) -> bool {
        matches!(self, Self::CustomSequence(_))
    }
}

/// Timing parameters for a stinger transition.
#[derive(Debug, Clone)]
pub struct TransitionTiming {
    /// Duration of pre-roll (time the outgoing scene remains before transition starts).
    pub pre_roll_ms: u32,
    /// Duration of the actual transition animation.
    pub transition_ms: u32,
    /// Duration of post-roll (time the incoming scene holds after transition).
    pub post_roll_ms: u32,
}

impl TransitionTiming {
    /// Create timing with the given durations.
    #[must_use]
    pub fn new(pre_roll_ms: u32, transition_ms: u32, post_roll_ms: u32) -> Self {
        Self {
            pre_roll_ms,
            transition_ms,
            post_roll_ms,
        }
    }

    /// Total wall-clock time of the entire transition sequence.
    #[must_use]
    pub fn total_ms(&self) -> u64 {
        u64::from(self.pre_roll_ms) + u64::from(self.transition_ms) + u64::from(self.post_roll_ms)
    }

    /// Compute the alpha blend value at a given position within the transition.
    /// Returns a value in [0.0, 1.0] where 0.0 = outgoing scene visible,
    /// 1.0 = incoming scene visible.
    #[must_use]
    pub fn alpha_at(&self, elapsed_ms: u32) -> f32 {
        if elapsed_ms <= self.pre_roll_ms {
            return 0.0;
        }
        let transition_elapsed = elapsed_ms.saturating_sub(self.pre_roll_ms);
        if self.transition_ms == 0 {
            return 1.0;
        }
        let ratio = transition_elapsed as f32 / self.transition_ms as f32;
        ratio.clamp(0.0, 1.0)
    }
}

impl Default for TransitionTiming {
    fn default() -> Self {
        Self {
            pre_roll_ms: 200,
            transition_ms: 500,
            post_roll_ms: 200,
        }
    }
}

/// A stinger transition between scenes.
#[derive(Debug, Clone)]
pub struct StingerTransition {
    /// The visual preset to use.
    pub preset: StingerPreset,
    /// Timing parameters.
    pub timing: TransitionTiming,
    /// Resolution of the transition overlay (width, height).
    pub resolution: (u32, u32),
    /// Whether to use alpha blending (otherwise hard cut at midpoint).
    pub use_alpha: bool,
}

impl StingerTransition {
    /// Create a new stinger transition.
    #[must_use]
    pub fn new(preset: StingerPreset, timing: TransitionTiming) -> Self {
        Self {
            preset,
            timing,
            resolution: (1920, 1080),
            use_alpha: true,
        }
    }

    /// Compute a transition frame at the given elapsed milliseconds.
    /// Returns the alpha value and the current phase.
    #[must_use]
    pub fn evaluate(&self, elapsed_ms: u32) -> TransitionFrame {
        let alpha = self.timing.alpha_at(elapsed_ms);
        let total = self.timing.total_ms() as u32;
        let phase = if elapsed_ms <= self.timing.pre_roll_ms {
            TransitionPhase::PreRoll
        } else if elapsed_ms <= self.timing.pre_roll_ms + self.timing.transition_ms {
            TransitionPhase::Transition
        } else if elapsed_ms <= total {
            TransitionPhase::PostRoll
        } else {
            TransitionPhase::Complete
        };

        TransitionFrame { alpha, phase }
    }
}

/// Current phase of a stinger transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionPhase {
    /// Before the transition starts.
    PreRoll,
    /// During the animated transition.
    Transition,
    /// After the transition finishes.
    PostRoll,
    /// Transition is fully complete.
    Complete,
}

/// A single evaluated transition state.
#[derive(Debug, Clone)]
pub struct TransitionFrame {
    /// Alpha blend value (0.0 = outgoing, 1.0 = incoming).
    pub alpha: f32,
    /// Which phase the transition is in.
    pub phase: TransitionPhase,
}

// ---------------------------------------------------------------------------
// ReplayMarker & EventTimeline
// ---------------------------------------------------------------------------

/// A marker on the event timeline indicating a noteworthy moment.
#[derive(Debug, Clone)]
pub struct ReplayMarker {
    /// Milliseconds since session start.
    pub timestamp_ms: u64,
    /// Label for the marker.
    pub label: String,
    /// Associated event type, if any.
    pub event_type: Option<GameEventType>,
    /// Importance level (higher = more important).
    pub importance: u8,
}

/// A timeline that tracks events and replay markers for a session.
#[derive(Debug, Default)]
pub struct EventTimeline {
    /// All replay markers.
    markers: Vec<ReplayMarker>,
    /// Associated event recorder.
    recorder: EventRecorder,
}

impl EventTimeline {
    /// Create a new event timeline.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a game event and optionally add a replay marker.
    pub fn record_event(&mut self, event: GameEvent, add_marker: bool, importance: u8) {
        let ts = event.timestamp_ms;
        let etype = event.event_type.clone();
        let label = etype.name();
        self.recorder.record(event);

        if add_marker {
            self.markers.push(ReplayMarker {
                timestamp_ms: ts,
                label,
                event_type: Some(etype),
                importance,
            });
        }
    }

    /// Add a standalone replay marker (not tied to a recorded event).
    pub fn add_marker(&mut self, timestamp_ms: u64, label: &str, importance: u8) {
        self.markers.push(ReplayMarker {
            timestamp_ms,
            label: label.to_string(),
            event_type: None,
            importance,
        });
    }

    /// Get all markers with importance >= threshold.
    #[must_use]
    pub fn markers_above(&self, threshold: u8) -> Vec<&ReplayMarker> {
        self.markers
            .iter()
            .filter(|m| m.importance >= threshold)
            .collect()
    }

    /// Get all markers in a time range.
    #[must_use]
    pub fn markers_in_range(&self, start_ms: u64, end_ms: u64) -> Vec<&ReplayMarker> {
        self.markers
            .iter()
            .filter(|m| m.timestamp_ms >= start_ms && m.timestamp_ms <= end_ms)
            .collect()
    }

    /// Total number of markers.
    #[must_use]
    pub fn marker_count(&self) -> usize {
        self.markers.len()
    }

    /// Total number of recorded events.
    #[must_use]
    pub fn event_count(&self) -> usize {
        self.recorder.event_count()
    }

    /// Access the underlying recorder.
    #[must_use]
    pub fn recorder(&self) -> &EventRecorder {
        &self.recorder
    }

    /// Clear all markers and events.
    pub fn clear(&mut self) {
        self.markers.clear();
        self.recorder.clear();
    }
}

// ---------------------------------------------------------------------------
// EventDetector -- heuristic kill/death/achievement detection
// ---------------------------------------------------------------------------

/// Configurable thresholds for heuristic event detection.
#[derive(Debug, Clone)]
pub struct DetectionThresholds {
    /// Minimum number of kills within `kill_window_ms` to trigger a kill-streak alert.
    pub kill_streak_count: u32,
    /// Time window (ms) in which kills are counted for a streak.
    pub kill_window_ms: u64,
    /// Maximum gap between death and next kill to count as a "revenge" event.
    pub revenge_window_ms: u64,
    /// Number of consecutive deaths within `death_window_ms` to trigger a "feed" alert.
    pub death_feed_count: u32,
    /// Time window (ms) for death-feed detection.
    pub death_window_ms: u64,
    /// Minimum time between achievements to avoid duplicates (debounce).
    pub achievement_debounce_ms: u64,
}

impl Default for DetectionThresholds {
    fn default() -> Self {
        Self {
            kill_streak_count: 3,
            kill_window_ms: 10_000,
            revenge_window_ms: 5_000,
            death_feed_count: 3,
            death_window_ms: 30_000,
            achievement_debounce_ms: 1_000,
        }
    }
}

/// Detected event from the heuristic analyzer.
#[derive(Debug, Clone, PartialEq)]
pub enum DetectedEvent {
    /// A kill streak of the given length was detected.
    KillStreak(u32),
    /// A revenge kill (kill shortly after being killed).
    RevengeKill,
    /// The player is feeding (dying too often).
    DeathFeed(u32),
    /// An achievement was unlocked.
    Achievement,
}

/// Heuristic event detector that analyzes a stream of game events and
/// detects higher-level patterns (kill streaks, revenge kills, etc.).
#[derive(Debug)]
pub struct EventDetector {
    thresholds: DetectionThresholds,
    recent_kills: Vec<u64>,
    recent_deaths: Vec<u64>,
    last_death_ts: Option<u64>,
    last_achievement_ts: Option<u64>,
}

impl EventDetector {
    /// Create a new detector with the given thresholds.
    #[must_use]
    pub fn new(thresholds: DetectionThresholds) -> Self {
        Self {
            thresholds,
            recent_kills: Vec::new(),
            recent_deaths: Vec::new(),
            last_death_ts: None,
            last_achievement_ts: None,
        }
    }

    /// Create a detector with default thresholds.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(DetectionThresholds::default())
    }

    /// Feed an event to the detector and get any detected higher-level events.
    pub fn process(&mut self, event: &GameEvent) -> Vec<DetectedEvent> {
        let ts = event.timestamp_ms;
        let mut detected = Vec::new();

        match &event.event_type {
            GameEventType::Kill => {
                // Clean old kills outside window
                let window_start = ts.saturating_sub(self.thresholds.kill_window_ms);
                self.recent_kills.retain(|&t| t >= window_start);
                self.recent_kills.push(ts);

                // Check kill streak
                if self.recent_kills.len() as u32 >= self.thresholds.kill_streak_count {
                    detected.push(DetectedEvent::KillStreak(self.recent_kills.len() as u32));
                }

                // Check revenge
                if let Some(death_ts) = self.last_death_ts {
                    if ts.saturating_sub(death_ts) <= self.thresholds.revenge_window_ms {
                        detected.push(DetectedEvent::RevengeKill);
                    }
                }
            }
            GameEventType::Death => {
                let window_start = ts.saturating_sub(self.thresholds.death_window_ms);
                self.recent_deaths.retain(|&t| t >= window_start);
                self.recent_deaths.push(ts);
                self.last_death_ts = Some(ts);

                // Clear kill streak on death
                self.recent_kills.clear();

                // Check death feed
                if self.recent_deaths.len() as u32 >= self.thresholds.death_feed_count {
                    detected.push(DetectedEvent::DeathFeed(self.recent_deaths.len() as u32));
                }
            }
            GameEventType::Achievement => {
                let should_detect = match self.last_achievement_ts {
                    Some(last_ts) => {
                        ts.saturating_sub(last_ts) >= self.thresholds.achievement_debounce_ms
                    }
                    None => true,
                };
                if should_detect {
                    self.last_achievement_ts = Some(ts);
                    detected.push(DetectedEvent::Achievement);
                }
            }
            _ => {}
        }

        detected
    }

    /// Reset the detector state.
    pub fn reset(&mut self) {
        self.recent_kills.clear();
        self.recent_deaths.clear();
        self.last_death_ts = None;
        self.last_achievement_ts = None;
    }

    /// Get current thresholds.
    #[must_use]
    pub fn thresholds(&self) -> &DetectionThresholds {
        &self.thresholds
    }

    /// Update thresholds.
    pub fn set_thresholds(&mut self, thresholds: DetectionThresholds) {
        self.thresholds = thresholds;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn evt(et: GameEventType, ts: u64) -> GameEvent {
        GameEvent::new(et, ts)
    }

    // --- GameEventType ---

    #[test]
    fn test_event_type_name_kill() {
        assert_eq!(GameEventType::Kill.name(), "Kill");
    }

    #[test]
    fn test_event_type_name_death() {
        assert_eq!(GameEventType::Death.name(), "Death");
    }

    #[test]
    fn test_event_type_name_pickup() {
        assert_eq!(GameEventType::Pickup.name(), "Pickup");
    }

    #[test]
    fn test_event_type_name_respawn() {
        assert_eq!(GameEventType::Respawn.name(), "Respawn");
    }

    #[test]
    fn test_event_type_name_level_start() {
        assert_eq!(GameEventType::LevelStart.name(), "LevelStart");
    }

    #[test]
    fn test_event_type_name_level_end() {
        assert_eq!(GameEventType::LevelEnd.name(), "LevelEnd");
    }

    #[test]
    fn test_event_type_name_achievement() {
        assert_eq!(GameEventType::Achievement.name(), "Achievement");
    }

    #[test]
    fn test_event_type_name_custom() {
        assert_eq!(
            GameEventType::Custom("BossKill".to_string()).name(),
            "BossKill"
        );
    }

    #[test]
    fn test_same_kind_matching() {
        assert!(GameEventType::Kill.same_kind(&GameEventType::Kill));
        assert!(GameEventType::Custom("a".to_string())
            .same_kind(&GameEventType::Custom("b".to_string())));
    }

    #[test]
    fn test_same_kind_non_matching() {
        assert!(!GameEventType::Kill.same_kind(&GameEventType::Death));
    }

    // --- GameEvent ---

    #[test]
    fn test_game_event_with_data() {
        let e = GameEvent::new(GameEventType::Kill, 1000)
            .with_data("weapon", "sword")
            .with_data("player", "hero");
        assert_eq!(e.get_data("weapon"), Some(&"sword".to_string()));
        assert_eq!(e.get_data("player"), Some(&"hero".to_string()));
        assert!(e.get_data("missing").is_none());
    }

    // --- EventRecorder basic ---

    #[test]
    fn test_recorder_empty_initially() {
        let rec = EventRecorder::new();
        assert_eq!(rec.event_count(), 0);
        assert!(rec.latest_event().is_none());
    }

    #[test]
    fn test_record_and_count() {
        let mut rec = EventRecorder::new();
        rec.record(evt(GameEventType::Kill, 100));
        rec.record(evt(GameEventType::Death, 200));
        assert_eq!(rec.event_count(), 2);
    }

    #[test]
    fn test_clear() {
        let mut rec = EventRecorder::new();
        rec.record(evt(GameEventType::Kill, 100));
        rec.clear();
        assert_eq!(rec.event_count(), 0);
    }

    #[test]
    fn test_latest_event() {
        let mut rec = EventRecorder::new();
        rec.record(evt(GameEventType::Kill, 100));
        rec.record(evt(GameEventType::Achievement, 500));
        let latest = rec.latest_event().expect("should have latest");
        assert_eq!(latest.event_type, GameEventType::Achievement);
        assert_eq!(latest.timestamp_ms, 500);
    }

    // --- events_of_type ---

    #[test]
    fn test_events_of_type_kill() {
        let mut rec = EventRecorder::new();
        rec.record(evt(GameEventType::Kill, 100));
        rec.record(evt(GameEventType::Death, 200));
        rec.record(evt(GameEventType::Kill, 300));
        assert_eq!(rec.events_of_type(&GameEventType::Kill).len(), 2);
        assert_eq!(rec.events_of_type(&GameEventType::Death).len(), 1);
        assert_eq!(rec.events_of_type(&GameEventType::Pickup).len(), 0);
    }

    #[test]
    fn test_events_of_type_custom_any_label() {
        let mut rec = EventRecorder::new();
        rec.record(evt(GameEventType::Custom("BossKill".to_string()), 100));
        rec.record(evt(GameEventType::Custom("SecretFound".to_string()), 200));
        rec.record(evt(GameEventType::Kill, 300));
        // Any Custom matches Custom query
        let customs = rec.events_of_type(&GameEventType::Custom("anything".to_string()));
        assert_eq!(customs.len(), 2);
    }

    // --- events_in_range ---

    #[test]
    fn test_events_in_range() {
        let mut rec = EventRecorder::new();
        rec.record(evt(GameEventType::Kill, 100));
        rec.record(evt(GameEventType::Kill, 500));
        rec.record(evt(GameEventType::Kill, 1000));
        let in_range = rec.events_in_range(200, 800);
        assert_eq!(in_range.len(), 1);
        assert_eq!(in_range[0].timestamp_ms, 500);
    }

    #[test]
    fn test_events_in_range_inclusive_bounds() {
        let mut rec = EventRecorder::new();
        rec.record(evt(GameEventType::Kill, 100));
        rec.record(evt(GameEventType::Kill, 1000));
        let in_range = rec.events_in_range(100, 1000);
        assert_eq!(in_range.len(), 2);
    }

    // --- export JSON ---

    #[test]
    fn test_export_json_empty() {
        let rec = EventRecorder::new();
        assert_eq!(rec.export_events(EventFormat::Json), "[]");
    }

    #[test]
    fn test_export_json_contains_type_and_timestamp() {
        let mut rec = EventRecorder::new();
        rec.record(evt(GameEventType::Kill, 1234));
        let json = rec.export_events(EventFormat::Json);
        assert!(json.contains("\"type\":\"Kill\""));
        assert!(json.contains("\"timestamp_ms\":1234"));
    }

    #[test]
    fn test_export_json_with_data() {
        let mut rec = EventRecorder::new();
        rec.record(GameEvent::new(GameEventType::Pickup, 50).with_data("item", "health_pack"));
        let json = rec.export_events(EventFormat::Json);
        assert!(json.contains("\"item\":\"health_pack\""));
    }

    // --- export CSV ---

    #[test]
    fn test_export_csv_has_header() {
        let rec = EventRecorder::new();
        let csv = rec.export_events(EventFormat::Csv);
        assert!(csv.starts_with("type,timestamp_ms,data\n"));
    }

    #[test]
    fn test_export_csv_row_content() {
        let mut rec = EventRecorder::new();
        rec.record(evt(GameEventType::Death, 999));
        let csv = rec.export_events(EventFormat::Csv);
        assert!(csv.contains("Death,999,"));
    }

    // --- export GameSpecific ---

    #[test]
    fn test_export_game_specific_format() {
        let mut rec = EventRecorder::new();
        rec.record(evt(GameEventType::Respawn, 42));
        let gs = rec.export_events(EventFormat::GameSpecific);
        assert!(gs.contains("[42] Respawn:"));
    }

    #[test]
    fn test_export_game_specific_with_data() {
        let mut rec = EventRecorder::new();
        rec.record(GameEvent::new(GameEventType::Achievement, 777).with_data("name", "speedrun"));
        let gs = rec.export_events(EventFormat::GameSpecific);
        assert!(gs.contains("[777] Achievement:"));
        assert!(gs.contains("name=speedrun"));
    }

    // -- StingerPreset --

    #[test]
    fn test_stinger_preset_names() {
        assert_eq!(StingerPreset::SlideLeft.name(), "SlideLeft");
        assert_eq!(StingerPreset::SlideRight.name(), "SlideRight");
        assert_eq!(StingerPreset::WipeDown.name(), "WipeDown");
        assert_eq!(StingerPreset::WipeUp.name(), "WipeUp");
        assert_eq!(StingerPreset::Fade.name(), "Fade");
        assert_eq!(StingerPreset::Zoom.name(), "Zoom");
        assert_eq!(
            StingerPreset::CustomSequence(vec![]).name(),
            "CustomSequence"
        );
    }

    #[test]
    fn test_stinger_preset_is_custom() {
        assert!(!StingerPreset::Fade.is_custom());
        assert!(StingerPreset::CustomSequence(vec![vec![1, 2, 3]]).is_custom());
    }

    // -- TransitionTiming --

    #[test]
    fn test_transition_timing_total() {
        let timing = TransitionTiming::new(200, 500, 300);
        assert_eq!(timing.total_ms(), 1000);
    }

    #[test]
    fn test_transition_timing_alpha_pre_roll() {
        let timing = TransitionTiming::new(200, 500, 200);
        assert!((timing.alpha_at(0) - 0.0).abs() < f32::EPSILON);
        assert!((timing.alpha_at(100) - 0.0).abs() < f32::EPSILON);
        assert!((timing.alpha_at(200) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_transition_timing_alpha_mid_transition() {
        let timing = TransitionTiming::new(0, 1000, 0);
        let alpha = timing.alpha_at(500);
        assert!((alpha - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_transition_timing_alpha_post_roll() {
        let timing = TransitionTiming::new(100, 200, 100);
        let alpha = timing.alpha_at(400);
        assert!((alpha - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_transition_timing_zero_duration() {
        let timing = TransitionTiming::new(0, 0, 0);
        assert!((timing.alpha_at(0) - 0.0).abs() < f32::EPSILON);
        assert!((timing.alpha_at(1) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_transition_timing_default() {
        let timing = TransitionTiming::default();
        assert_eq!(timing.pre_roll_ms, 200);
        assert_eq!(timing.transition_ms, 500);
        assert_eq!(timing.post_roll_ms, 200);
    }

    // -- StingerTransition --

    #[test]
    fn test_stinger_evaluate_phases() {
        let stinger =
            StingerTransition::new(StingerPreset::Fade, TransitionTiming::new(100, 200, 100));

        let f0 = stinger.evaluate(50);
        assert_eq!(f0.phase, TransitionPhase::PreRoll);
        assert!((f0.alpha - 0.0).abs() < f32::EPSILON);

        let f1 = stinger.evaluate(200);
        assert_eq!(f1.phase, TransitionPhase::Transition);
        assert!(f1.alpha >= 0.0 && f1.alpha <= 1.0);

        let f2 = stinger.evaluate(350);
        assert_eq!(f2.phase, TransitionPhase::PostRoll);

        let f3 = stinger.evaluate(500);
        assert_eq!(f3.phase, TransitionPhase::Complete);
    }

    // -- EventTimeline --

    #[test]
    fn test_event_timeline_empty() {
        let tl = EventTimeline::new();
        assert_eq!(tl.marker_count(), 0);
        assert_eq!(tl.event_count(), 0);
    }

    #[test]
    fn test_event_timeline_record_with_marker() {
        let mut tl = EventTimeline::new();
        tl.record_event(evt(GameEventType::Kill, 1000), true, 5);
        assert_eq!(tl.event_count(), 1);
        assert_eq!(tl.marker_count(), 1);
    }

    #[test]
    fn test_event_timeline_record_without_marker() {
        let mut tl = EventTimeline::new();
        tl.record_event(evt(GameEventType::Kill, 1000), false, 5);
        assert_eq!(tl.event_count(), 1);
        assert_eq!(tl.marker_count(), 0);
    }

    #[test]
    fn test_event_timeline_add_standalone_marker() {
        let mut tl = EventTimeline::new();
        tl.add_marker(5000, "Epic Moment", 10);
        assert_eq!(tl.marker_count(), 1);
        assert_eq!(tl.event_count(), 0);
    }

    #[test]
    fn test_event_timeline_markers_above_threshold() {
        let mut tl = EventTimeline::new();
        tl.record_event(evt(GameEventType::Kill, 100), true, 3);
        tl.record_event(evt(GameEventType::Achievement, 200), true, 8);
        tl.add_marker(300, "custom", 10);

        assert_eq!(tl.markers_above(5).len(), 2);
        assert_eq!(tl.markers_above(9).len(), 1);
        assert_eq!(tl.markers_above(1).len(), 3);
    }

    #[test]
    fn test_event_timeline_markers_in_range() {
        let mut tl = EventTimeline::new();
        tl.add_marker(100, "a", 1);
        tl.add_marker(500, "b", 1);
        tl.add_marker(1000, "c", 1);
        assert_eq!(tl.markers_in_range(200, 800).len(), 1);
        assert_eq!(tl.markers_in_range(100, 1000).len(), 3);
    }

    #[test]
    fn test_event_timeline_clear() {
        let mut tl = EventTimeline::new();
        tl.record_event(evt(GameEventType::Kill, 100), true, 5);
        tl.add_marker(200, "x", 5);
        tl.clear();
        assert_eq!(tl.event_count(), 0);
        assert_eq!(tl.marker_count(), 0);
    }

    // -- EventDetector --

    #[test]
    fn test_detector_kill_streak() {
        let mut det = EventDetector::new(DetectionThresholds {
            kill_streak_count: 3,
            kill_window_ms: 10_000,
            ..DetectionThresholds::default()
        });

        let r1 = det.process(&evt(GameEventType::Kill, 1000));
        assert!(r1.is_empty());
        let r2 = det.process(&evt(GameEventType::Kill, 2000));
        assert!(r2.is_empty());
        let r3 = det.process(&evt(GameEventType::Kill, 3000));
        assert!(r3.contains(&DetectedEvent::KillStreak(3)));
    }

    #[test]
    fn test_detector_revenge_kill() {
        let mut det = EventDetector::new(DetectionThresholds {
            revenge_window_ms: 5000,
            kill_streak_count: 100, // disable kill streak detection
            ..DetectionThresholds::default()
        });

        det.process(&evt(GameEventType::Death, 1000));
        let result = det.process(&evt(GameEventType::Kill, 3000));
        assert!(result.contains(&DetectedEvent::RevengeKill));
    }

    #[test]
    fn test_detector_no_revenge_after_window() {
        let mut det = EventDetector::new(DetectionThresholds {
            revenge_window_ms: 2000,
            kill_streak_count: 100,
            ..DetectionThresholds::default()
        });

        det.process(&evt(GameEventType::Death, 1000));
        let result = det.process(&evt(GameEventType::Kill, 5000));
        assert!(!result.contains(&DetectedEvent::RevengeKill));
    }

    #[test]
    fn test_detector_death_feed() {
        let mut det = EventDetector::new(DetectionThresholds {
            death_feed_count: 3,
            death_window_ms: 30_000,
            ..DetectionThresholds::default()
        });

        det.process(&evt(GameEventType::Death, 1000));
        det.process(&evt(GameEventType::Death, 2000));
        let r = det.process(&evt(GameEventType::Death, 3000));
        assert!(r.contains(&DetectedEvent::DeathFeed(3)));
    }

    #[test]
    fn test_detector_death_clears_kill_streak() {
        let mut det = EventDetector::new(DetectionThresholds {
            kill_streak_count: 3,
            kill_window_ms: 10_000,
            ..DetectionThresholds::default()
        });

        det.process(&evt(GameEventType::Kill, 1000));
        det.process(&evt(GameEventType::Kill, 2000));
        det.process(&evt(GameEventType::Death, 3000));
        // After death, kill streak resets -- next kill should not trigger streak
        let r = det.process(&evt(GameEventType::Kill, 4000));
        assert!(!r.iter().any(|e| matches!(e, DetectedEvent::KillStreak(_))));
    }

    #[test]
    fn test_detector_achievement_debounce() {
        let mut det = EventDetector::new(DetectionThresholds {
            achievement_debounce_ms: 2000,
            ..DetectionThresholds::default()
        });

        let r1 = det.process(&evt(GameEventType::Achievement, 1000));
        assert!(r1.contains(&DetectedEvent::Achievement));

        // Too soon -- debounced
        let r2 = det.process(&evt(GameEventType::Achievement, 2000));
        assert!(!r2.contains(&DetectedEvent::Achievement));

        // After debounce window
        let r3 = det.process(&evt(GameEventType::Achievement, 3500));
        assert!(r3.contains(&DetectedEvent::Achievement));
    }

    #[test]
    fn test_detector_reset() {
        let mut det = EventDetector::with_defaults();
        det.process(&evt(GameEventType::Kill, 1000));
        det.process(&evt(GameEventType::Death, 2000));
        det.reset();

        // After reset, kill streak count should start from 0
        let r = det.process(&evt(GameEventType::Kill, 5000));
        assert!(!r.iter().any(|e| matches!(e, DetectedEvent::KillStreak(_))));
    }

    #[test]
    fn test_detector_set_thresholds() {
        let mut det = EventDetector::with_defaults();
        assert_eq!(det.thresholds().kill_streak_count, 3);
        det.set_thresholds(DetectionThresholds {
            kill_streak_count: 5,
            ..DetectionThresholds::default()
        });
        assert_eq!(det.thresholds().kill_streak_count, 5);
    }

    #[test]
    fn test_detector_ignores_other_events() {
        let mut det = EventDetector::with_defaults();
        let r = det.process(&evt(GameEventType::Pickup, 1000));
        assert!(r.is_empty());
        let r2 = det.process(&evt(GameEventType::LevelStart, 2000));
        assert!(r2.is_empty());
    }
}
