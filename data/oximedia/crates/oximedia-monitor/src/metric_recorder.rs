//! Metric recording and playback for debugging.
//!
//! Records live metric streams to an in-memory journal that can be serialized
//! to JSON for offline replay. This is invaluable for debugging intermittent
//! monitoring issues: capture the exact metric sequence that triggered a
//! problem, then replay it deterministically.
//!
//! # Recording
//!
//! ```ignore
//! let mut recorder = MetricRecorder::new(RecorderConfig::default());
//! recorder.start_recording("debug_session_1");
//! recorder.record("cpu_usage", 85.0);
//! recorder.record("memory_usage", 72.3);
//! let journal = recorder.stop_recording().unwrap();
//! let json = journal.to_json().unwrap();
//! ```
//!
//! # Playback
//!
//! ```ignore
//! let journal = RecordingJournal::from_json(&json).unwrap();
//! let mut player = MetricPlayer::new(journal);
//! while let Some(entry) = player.next_entry() {
//!     println!("{}: {} = {}", entry.offset_ms, entry.metric, entry.value);
//! }
//! ```

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::error::{MonitorError, MonitorResult};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the metric recorder.
#[derive(Debug, Clone)]
pub struct RecorderConfig {
    /// Maximum number of entries per recording session.
    pub max_entries: usize,
    /// Maximum recording duration.
    pub max_duration: Duration,
    /// Whether to capture metric labels/tags.
    pub capture_labels: bool,
}

impl Default for RecorderConfig {
    fn default() -> Self {
        Self {
            max_entries: 100_000,
            max_duration: Duration::from_secs(3600), // 1 hour
            capture_labels: true,
        }
    }
}

impl RecorderConfig {
    /// Set the max entries.
    #[must_use]
    pub fn with_max_entries(mut self, n: usize) -> Self {
        self.max_entries = n.max(1);
        self
    }

    /// Set the max duration.
    #[must_use]
    pub fn with_max_duration(mut self, d: Duration) -> Self {
        self.max_duration = d;
        self
    }
}

// ---------------------------------------------------------------------------
// Recording journal
// ---------------------------------------------------------------------------

/// A single recorded metric entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordedEntry {
    /// Milliseconds elapsed since the recording session started.
    pub offset_ms: u64,
    /// Metric name.
    pub metric: String,
    /// Metric value.
    pub value: f64,
    /// Optional label key-value pairs.
    pub labels: HashMap<String, String>,
}

/// Metadata about a recording session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMetadata {
    /// Session identifier.
    pub session_id: String,
    /// ISO-8601 timestamp when recording started.
    pub started_at: String,
    /// Total duration of the recording in milliseconds.
    pub duration_ms: u64,
    /// Number of entries recorded.
    pub entry_count: usize,
    /// Number of distinct metrics recorded.
    pub metric_count: usize,
    /// Reason the recording stopped.
    pub stop_reason: StopReason,
}

/// Why the recording was stopped.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum StopReason {
    /// Manually stopped by the user.
    Manual,
    /// Hit the maximum entry count.
    MaxEntries,
    /// Hit the maximum duration.
    MaxDuration,
}

/// A complete recorded journal containing metadata and entries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordingJournal {
    /// Session metadata.
    pub metadata: SessionMetadata,
    /// Ordered list of recorded entries.
    pub entries: Vec<RecordedEntry>,
}

impl RecordingJournal {
    /// Serialize to JSON.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    pub fn to_json(&self) -> MonitorResult<String> {
        serde_json::to_string_pretty(self).map_err(MonitorError::Serialization)
    }

    /// Serialize to compact JSON (no whitespace).
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    pub fn to_json_compact(&self) -> MonitorResult<String> {
        serde_json::to_string(self).map_err(MonitorError::Serialization)
    }

    /// Deserialize from JSON.
    ///
    /// # Errors
    ///
    /// Returns an error if deserialization fails.
    pub fn from_json(json: &str) -> MonitorResult<Self> {
        serde_json::from_str(json).map_err(MonitorError::Serialization)
    }

    /// Get all distinct metric names in the journal.
    #[must_use]
    pub fn metric_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .entries
            .iter()
            .map(|e| e.metric.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        names.sort();
        names
    }

    /// Filter entries by metric name.
    #[must_use]
    pub fn entries_for(&self, metric: &str) -> Vec<&RecordedEntry> {
        self.entries.iter().filter(|e| e.metric == metric).collect()
    }

    /// Get the time range of entries (first offset, last offset) in ms.
    #[must_use]
    pub fn time_range_ms(&self) -> Option<(u64, u64)> {
        if self.entries.is_empty() {
            return None;
        }
        let first = self.entries.first().map(|e| e.offset_ms)?;
        let last = self.entries.last().map(|e| e.offset_ms)?;
        Some((first, last))
    }
}

// ---------------------------------------------------------------------------
// Recorder
// ---------------------------------------------------------------------------

/// Recording state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecorderState {
    /// Not recording.
    Idle,
    /// Actively recording.
    Recording,
}

/// Metric recorder that captures live metric streams.
#[derive(Debug)]
pub struct MetricRecorder {
    config: RecorderConfig,
    state: RecorderState,
    session_id: String,
    start_time: Option<Instant>,
    start_iso: String,
    entries: Vec<RecordedEntry>,
    metric_names_seen: std::collections::HashSet<String>,
    stop_reason: Option<StopReason>,
}

impl MetricRecorder {
    /// Create a new recorder.
    #[must_use]
    pub fn new(config: RecorderConfig) -> Self {
        Self {
            config,
            state: RecorderState::Idle,
            session_id: String::new(),
            start_time: None,
            start_iso: String::new(),
            entries: Vec::new(),
            metric_names_seen: std::collections::HashSet::new(),
            stop_reason: None,
        }
    }

    /// Create a recorder with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(RecorderConfig::default())
    }

    /// Start a recording session.
    ///
    /// # Errors
    ///
    /// Returns an error if already recording.
    pub fn start_recording(&mut self, session_id: impl Into<String>) -> MonitorResult<()> {
        if self.state == RecorderState::Recording {
            return Err(MonitorError::Other(
                "Already recording; stop the current session first".to_string(),
            ));
        }

        self.session_id = session_id.into();
        self.state = RecorderState::Recording;
        self.start_time = Some(Instant::now());
        self.start_iso = chrono::Utc::now().to_rfc3339();
        self.entries.clear();
        self.metric_names_seen.clear();
        self.stop_reason = None;

        Ok(())
    }

    /// Record a metric value.
    ///
    /// If not currently recording, this is a no-op.
    /// Returns `true` if the entry was recorded, `false` if skipped.
    pub fn record(&mut self, metric: &str, value: f64) -> bool {
        self.record_with_labels(metric, value, HashMap::new())
    }

    /// Record a metric value with labels.
    ///
    /// Returns `true` if the entry was recorded.
    pub fn record_with_labels(
        &mut self,
        metric: &str,
        value: f64,
        labels: HashMap<String, String>,
    ) -> bool {
        if self.state != RecorderState::Recording {
            return false;
        }

        let start = match self.start_time {
            Some(t) => t,
            None => return false,
        };

        // Check duration limit.
        let elapsed = start.elapsed();
        if elapsed > self.config.max_duration {
            self.stop_reason = Some(StopReason::MaxDuration);
            self.state = RecorderState::Idle;
            return false;
        }

        // Check entry limit.
        if self.entries.len() >= self.config.max_entries {
            self.stop_reason = Some(StopReason::MaxEntries);
            self.state = RecorderState::Idle;
            return false;
        }

        let offset_ms = elapsed.as_millis() as u64;
        self.metric_names_seen.insert(metric.to_string());

        let labels_to_store = if self.config.capture_labels {
            labels
        } else {
            HashMap::new()
        };

        self.entries.push(RecordedEntry {
            offset_ms,
            metric: metric.to_string(),
            value,
            labels: labels_to_store,
        });

        true
    }

    /// Stop recording and return the journal.
    ///
    /// # Errors
    ///
    /// Returns an error if not currently recording (and no auto-stopped data).
    pub fn stop_recording(&mut self) -> MonitorResult<RecordingJournal> {
        let reason = self.stop_reason.take().unwrap_or(StopReason::Manual);

        if self.state == RecorderState::Idle && self.entries.is_empty() {
            return Err(MonitorError::Other(
                "Not currently recording and no data available".to_string(),
            ));
        }

        self.state = RecorderState::Idle;

        let duration_ms = self
            .start_time
            .map_or(0, |t| t.elapsed().as_millis() as u64);

        let metadata = SessionMetadata {
            session_id: self.session_id.clone(),
            started_at: self.start_iso.clone(),
            duration_ms,
            entry_count: self.entries.len(),
            metric_count: self.metric_names_seen.len(),
            stop_reason: reason,
        };

        let journal = RecordingJournal {
            metadata,
            entries: std::mem::take(&mut self.entries),
        };

        self.metric_names_seen.clear();

        Ok(journal)
    }

    /// Check if currently recording.
    #[must_use]
    pub fn is_recording(&self) -> bool {
        self.state == RecorderState::Recording
    }

    /// Get the current state.
    #[must_use]
    pub fn state(&self) -> RecorderState {
        self.state
    }

    /// Number of entries recorded so far in the current session.
    #[must_use]
    pub fn current_entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Reference to the configuration.
    #[must_use]
    pub fn config(&self) -> &RecorderConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Player
// ---------------------------------------------------------------------------

/// Playback speed multiplier.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PlaybackSpeed {
    /// Instant playback (no delays).
    Instant,
    /// Real-time playback (1x speed).
    RealTime,
    /// Custom multiplier (e.g. 2.0 = double speed, 0.5 = half speed).
    Multiplied(f64),
}

impl PlaybackSpeed {
    /// Convert an offset in ms to the actual delay in ms.
    #[must_use]
    pub fn adjust_ms(self, offset_ms: u64) -> u64 {
        match self {
            Self::Instant => 0,
            Self::RealTime => offset_ms,
            Self::Multiplied(factor) => {
                if factor <= 0.0 {
                    0
                } else {
                    (offset_ms as f64 / factor) as u64
                }
            }
        }
    }
}

/// Playback event emitted during replay.
#[derive(Debug, Clone)]
pub struct PlaybackEvent {
    /// The recorded entry being replayed.
    pub entry: RecordedEntry,
    /// Delay in milliseconds before this event (adjusted by speed).
    pub delay_ms: u64,
    /// 0-based index of this event in the journal.
    pub index: usize,
    /// Total entries in the journal.
    pub total: usize,
}

/// Metric player for replaying recorded journals.
#[derive(Debug)]
pub struct MetricPlayer {
    journal: RecordingJournal,
    speed: PlaybackSpeed,
    position: usize,
    /// Optional filter: only replay entries for these metrics.
    metric_filter: Option<std::collections::HashSet<String>>,
}

impl MetricPlayer {
    /// Create a new player for the given journal.
    #[must_use]
    pub fn new(journal: RecordingJournal) -> Self {
        Self {
            journal,
            speed: PlaybackSpeed::Instant,
            position: 0,
            metric_filter: None,
        }
    }

    /// Set the playback speed.
    #[must_use]
    pub fn with_speed(mut self, speed: PlaybackSpeed) -> Self {
        self.speed = speed;
        self
    }

    /// Set a metric filter (only replay these metrics).
    #[must_use]
    pub fn with_filter(mut self, metrics: Vec<String>) -> Self {
        self.metric_filter = Some(metrics.into_iter().collect());
        self
    }

    /// Reset to the beginning of the journal.
    pub fn reset(&mut self) {
        self.position = 0;
    }

    /// Returns `true` if there are more entries to replay.
    #[must_use]
    pub fn has_next(&self) -> bool {
        self.peek_next_index().is_some()
    }

    /// Peek the index of the next entry (respecting filter).
    fn peek_next_index(&self) -> Option<usize> {
        let entries = &self.journal.entries;
        let mut idx = self.position;
        while idx < entries.len() {
            if let Some(ref filter) = self.metric_filter {
                if filter.contains(&entries[idx].metric) {
                    return Some(idx);
                }
            } else {
                return Some(idx);
            }
            idx += 1;
        }
        None
    }

    /// Get the next playback event.
    #[must_use]
    pub fn next_event(&mut self) -> Option<PlaybackEvent> {
        let idx = self.peek_next_index()?;
        let entry = self.journal.entries[idx].clone();
        let total = self.journal.entries.len();

        // Calculate delay from previous entry.
        let delay_ms = if idx > 0 {
            let prev_offset = self.journal.entries[idx - 1].offset_ms;
            let diff = entry.offset_ms.saturating_sub(prev_offset);
            self.speed.adjust_ms(diff)
        } else {
            0
        };

        self.position = idx + 1;

        Some(PlaybackEvent {
            entry,
            delay_ms,
            index: idx,
            total,
        })
    }

    /// Collect all remaining events into a vector.
    pub fn collect_remaining(&mut self) -> Vec<PlaybackEvent> {
        let mut events = Vec::new();
        while let Some(event) = self.next_event() {
            events.push(event);
        }
        events
    }

    /// Current playback position (0-based index).
    #[must_use]
    pub fn position(&self) -> usize {
        self.position
    }

    /// Total number of entries in the journal.
    #[must_use]
    pub fn total_entries(&self) -> usize {
        self.journal.entries.len()
    }

    /// Progress as a fraction (0.0 to 1.0).
    #[must_use]
    pub fn progress(&self) -> f64 {
        if self.journal.entries.is_empty() {
            return 1.0;
        }
        self.position as f64 / self.journal.entries.len() as f64
    }

    /// Reference to the session metadata.
    #[must_use]
    pub fn metadata(&self) -> &SessionMetadata {
        &self.journal.metadata
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_journal(entries: Vec<RecordedEntry>) -> RecordingJournal {
        let metric_count = entries
            .iter()
            .map(|e| e.metric.as_str())
            .collect::<std::collections::HashSet<_>>()
            .len();
        RecordingJournal {
            metadata: SessionMetadata {
                session_id: "test".to_string(),
                started_at: "2024-01-01T00:00:00Z".to_string(),
                duration_ms: entries.last().map_or(0, |e| e.offset_ms),
                entry_count: entries.len(),
                metric_count,
                stop_reason: StopReason::Manual,
            },
            entries,
        }
    }

    fn sample_entries() -> Vec<RecordedEntry> {
        vec![
            RecordedEntry {
                offset_ms: 0,
                metric: "cpu".to_string(),
                value: 50.0,
                labels: HashMap::new(),
            },
            RecordedEntry {
                offset_ms: 100,
                metric: "memory".to_string(),
                value: 70.0,
                labels: HashMap::new(),
            },
            RecordedEntry {
                offset_ms: 200,
                metric: "cpu".to_string(),
                value: 55.0,
                labels: HashMap::new(),
            },
            RecordedEntry {
                offset_ms: 300,
                metric: "disk".to_string(),
                value: 80.0,
                labels: HashMap::new(),
            },
        ]
    }

    // -- RecorderConfig --

    #[test]
    fn test_recorder_config_default() {
        let cfg = RecorderConfig::default();
        assert_eq!(cfg.max_entries, 100_000);
        assert!(cfg.capture_labels);
    }

    #[test]
    fn test_recorder_config_builders() {
        let cfg = RecorderConfig::default()
            .with_max_entries(500)
            .with_max_duration(Duration::from_secs(60));
        assert_eq!(cfg.max_entries, 500);
        assert_eq!(cfg.max_duration, Duration::from_secs(60));
    }

    #[test]
    fn test_recorder_config_min_entries() {
        let cfg = RecorderConfig::default().with_max_entries(0);
        assert_eq!(cfg.max_entries, 1);
    }

    // -- MetricRecorder --

    #[test]
    fn test_recorder_starts_idle() {
        let r = MetricRecorder::with_defaults();
        assert_eq!(r.state(), RecorderState::Idle);
        assert!(!r.is_recording());
    }

    #[test]
    fn test_recorder_start_stop() {
        let mut r = MetricRecorder::with_defaults();
        r.start_recording("s1").expect("start should succeed");
        assert!(r.is_recording());
        r.record("cpu", 50.0);
        r.record("mem", 70.0);
        assert_eq!(r.current_entry_count(), 2);

        let journal = r.stop_recording().expect("stop should succeed");
        assert!(!r.is_recording());
        assert_eq!(journal.entries.len(), 2);
        assert_eq!(journal.metadata.session_id, "s1");
        assert_eq!(journal.metadata.stop_reason, StopReason::Manual);
    }

    #[test]
    fn test_recorder_double_start_error() {
        let mut r = MetricRecorder::with_defaults();
        r.start_recording("s1").expect("start should succeed");
        let result = r.start_recording("s2");
        assert!(result.is_err());
    }

    #[test]
    fn test_recorder_stop_without_start_error() {
        let mut r = MetricRecorder::with_defaults();
        let result = r.stop_recording();
        assert!(result.is_err());
    }

    #[test]
    fn test_recorder_record_while_idle_noop() {
        let mut r = MetricRecorder::with_defaults();
        assert!(!r.record("cpu", 50.0));
    }

    #[test]
    fn test_recorder_max_entries_auto_stop() {
        let mut r = MetricRecorder::new(RecorderConfig::default().with_max_entries(3));
        r.start_recording("s1").expect("start should succeed");
        assert!(r.record("a", 1.0));
        assert!(r.record("b", 2.0));
        assert!(r.record("c", 3.0));
        // Next record should hit the limit.
        assert!(!r.record("d", 4.0));

        let journal = r.stop_recording().expect("stop should succeed");
        assert_eq!(journal.entries.len(), 3);
        assert_eq!(journal.metadata.stop_reason, StopReason::MaxEntries);
    }

    #[test]
    fn test_recorder_with_labels() {
        let mut r = MetricRecorder::with_defaults();
        r.start_recording("s1").expect("start should succeed");
        let mut labels = HashMap::new();
        labels.insert("host".to_string(), "server1".to_string());
        r.record_with_labels("cpu", 50.0, labels);

        let journal = r.stop_recording().expect("stop should succeed");
        assert_eq!(
            journal.entries[0].labels.get("host").map(String::as_str),
            Some("server1")
        );
    }

    #[test]
    fn test_recorder_labels_not_captured_when_disabled() {
        let cfg = RecorderConfig {
            capture_labels: false,
            ..RecorderConfig::default()
        };
        let mut r = MetricRecorder::new(cfg);
        r.start_recording("s1").expect("start should succeed");
        let mut labels = HashMap::new();
        labels.insert("host".to_string(), "server1".to_string());
        r.record_with_labels("cpu", 50.0, labels);

        let journal = r.stop_recording().expect("stop should succeed");
        assert!(journal.entries[0].labels.is_empty());
    }

    // -- RecordingJournal --

    #[test]
    fn test_journal_to_json_roundtrip() {
        let journal = make_journal(sample_entries());
        let json = journal.to_json().expect("to_json should succeed");
        let restored = RecordingJournal::from_json(&json).expect("from_json should succeed");
        assert_eq!(restored.entries.len(), journal.entries.len());
        assert_eq!(restored.metadata.session_id, "test");
    }

    #[test]
    fn test_journal_compact_json() {
        let journal = make_journal(sample_entries());
        let json = journal.to_json_compact().expect("compact should succeed");
        assert!(!json.contains('\n'));
    }

    #[test]
    fn test_journal_metric_names() {
        let journal = make_journal(sample_entries());
        let names = journal.metric_names();
        assert_eq!(names, vec!["cpu", "disk", "memory"]);
    }

    #[test]
    fn test_journal_entries_for() {
        let journal = make_journal(sample_entries());
        let cpu_entries = journal.entries_for("cpu");
        assert_eq!(cpu_entries.len(), 2);
    }

    #[test]
    fn test_journal_time_range() {
        let journal = make_journal(sample_entries());
        let (start, end) = journal.time_range_ms().expect("range should exist");
        assert_eq!(start, 0);
        assert_eq!(end, 300);
    }

    #[test]
    fn test_journal_time_range_empty() {
        let journal = make_journal(Vec::new());
        assert!(journal.time_range_ms().is_none());
    }

    // -- PlaybackSpeed --

    #[test]
    fn test_playback_speed_instant() {
        assert_eq!(PlaybackSpeed::Instant.adjust_ms(1000), 0);
    }

    #[test]
    fn test_playback_speed_real_time() {
        assert_eq!(PlaybackSpeed::RealTime.adjust_ms(500), 500);
    }

    #[test]
    fn test_playback_speed_multiplied() {
        assert_eq!(PlaybackSpeed::Multiplied(2.0).adjust_ms(1000), 500);
        assert_eq!(PlaybackSpeed::Multiplied(0.5).adjust_ms(1000), 2000);
    }

    #[test]
    fn test_playback_speed_zero_multiplier() {
        assert_eq!(PlaybackSpeed::Multiplied(0.0).adjust_ms(1000), 0);
    }

    // -- MetricPlayer --

    #[test]
    fn test_player_basic_playback() {
        let journal = make_journal(sample_entries());
        let mut player = MetricPlayer::new(journal);
        assert!(player.has_next());
        assert_eq!(player.total_entries(), 4);

        let event = player.next_event().expect("event should exist");
        assert_eq!(event.entry.metric, "cpu");
        assert_eq!(event.index, 0);
        assert_eq!(event.delay_ms, 0); // first event has no delay
    }

    #[test]
    fn test_player_all_events() {
        let journal = make_journal(sample_entries());
        let mut player = MetricPlayer::new(journal);
        let events = player.collect_remaining();
        assert_eq!(events.len(), 4);
        assert!(!player.has_next());
    }

    #[test]
    fn test_player_delays_instant() {
        let journal = make_journal(sample_entries());
        let mut player = MetricPlayer::new(journal).with_speed(PlaybackSpeed::Instant);
        let events = player.collect_remaining();
        for event in &events {
            assert_eq!(event.delay_ms, 0);
        }
    }

    #[test]
    fn test_player_delays_real_time() {
        let journal = make_journal(sample_entries());
        let mut player = MetricPlayer::new(journal).with_speed(PlaybackSpeed::RealTime);
        let events = player.collect_remaining();
        // events[0] delay = 0, events[1] delay = 100, events[2] delay = 100, events[3] delay = 100
        assert_eq!(events[0].delay_ms, 0);
        assert_eq!(events[1].delay_ms, 100);
        assert_eq!(events[2].delay_ms, 100);
        assert_eq!(events[3].delay_ms, 100);
    }

    #[test]
    fn test_player_filter() {
        let journal = make_journal(sample_entries());
        let mut player = MetricPlayer::new(journal).with_filter(vec!["cpu".to_string()]);
        let events = player.collect_remaining();
        assert_eq!(events.len(), 2);
        assert!(events.iter().all(|e| e.entry.metric == "cpu"));
    }

    #[test]
    fn test_player_reset() {
        let journal = make_journal(sample_entries());
        let mut player = MetricPlayer::new(journal);
        let _ = player.next_event();
        let _ = player.next_event();
        assert_eq!(player.position(), 2);

        player.reset();
        assert_eq!(player.position(), 0);
        assert!(player.has_next());
    }

    #[test]
    fn test_player_progress() {
        let journal = make_journal(sample_entries());
        let mut player = MetricPlayer::new(journal);
        assert!((player.progress() - 0.0).abs() < 1e-9);
        let _ = player.next_event();
        let _ = player.next_event();
        assert!((player.progress() - 0.5).abs() < 1e-9);
        let _ = player.collect_remaining();
        assert!((player.progress() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_player_empty_journal() {
        let journal = make_journal(Vec::new());
        let player = MetricPlayer::new(journal);
        assert!(!player.has_next());
        assert!((player.progress() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_player_metadata() {
        let journal = make_journal(sample_entries());
        let player = MetricPlayer::new(journal);
        assert_eq!(player.metadata().session_id, "test");
    }

    // -- Full record-replay integration --

    #[test]
    fn test_record_serialize_replay() {
        let mut recorder = MetricRecorder::with_defaults();
        recorder
            .start_recording("integration_test")
            .expect("start should succeed");
        recorder.record("cpu", 50.0);
        recorder.record("memory", 70.0);
        recorder.record("cpu", 55.0);
        let journal = recorder.stop_recording().expect("stop should succeed");

        // Serialize.
        let json = journal.to_json().expect("to_json should succeed");

        // Deserialize.
        let restored = RecordingJournal::from_json(&json).expect("from_json should succeed");
        assert_eq!(restored.entries.len(), 3);

        // Replay.
        let mut player = MetricPlayer::new(restored);
        let events = player.collect_remaining();
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].entry.metric, "cpu");
        assert!((events[0].entry.value - 50.0).abs() < 1e-9);
        assert_eq!(events[2].entry.metric, "cpu");
        assert!((events[2].entry.value - 55.0).abs() < 1e-9);
    }

    #[test]
    fn test_record_multiple_sessions() {
        let mut recorder = MetricRecorder::with_defaults();

        // Session 1.
        recorder
            .start_recording("session1")
            .expect("start should succeed");
        recorder.record("cpu", 10.0);
        let j1 = recorder.stop_recording().expect("stop should succeed");
        assert_eq!(j1.metadata.session_id, "session1");

        // Session 2.
        recorder
            .start_recording("session2")
            .expect("start should succeed");
        recorder.record("mem", 20.0);
        recorder.record("mem", 30.0);
        let j2 = recorder.stop_recording().expect("stop should succeed");
        assert_eq!(j2.metadata.session_id, "session2");
        assert_eq!(j2.entries.len(), 2);
    }
}
