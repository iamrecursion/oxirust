//! Catch-up TV management: start-over, lookback windows, and recording triggers.

#![allow(dead_code)]

use std::collections::HashMap;
use std::time::{Duration, SystemTime};

// ── Lookback window ───────────────────────────────────────────────────────────

/// Configuration for the catch-up / lookback service
#[derive(Debug, Clone)]
pub struct CatchupConfig {
    /// Maximum lookback duration available to viewers
    pub lookback_window: Duration,
    /// Minimum remaining time before a programme expires from catch-up
    pub expiry_buffer: Duration,
    /// Maximum concurrent start-over streams per channel
    pub max_concurrent_startover: usize,
    /// Whether start-over is allowed even when recording is ongoing
    pub allow_startover_during_record: bool,
}

impl Default for CatchupConfig {
    fn default() -> Self {
        Self {
            lookback_window: Duration::from_hours(168), // 7 days
            expiry_buffer: Duration::from_hours(1),     // 1 hour
            max_concurrent_startover: 50,
            allow_startover_during_record: true,
        }
    }
}

// ── Recording state ───────────────────────────────────────────────────────────

/// State of a catch-up recording
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordingState {
    Scheduled,
    Recording,
    Completed,
    Failed,
    Expired,
}

impl RecordingState {
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Expired)
    }
}

/// A record of a broadcast programme available for catch-up
#[derive(Debug, Clone)]
pub struct CatchupRecording {
    pub id: String,
    pub channel_id: String,
    pub title: String,
    pub state: RecordingState,
    pub air_time: SystemTime,
    pub duration: Duration,
    pub file_path: Option<String>,
    pub byte_size: Option<u64>,
}

impl CatchupRecording {
    /// Create a new scheduled recording
    pub fn new(
        id: &str,
        channel_id: &str,
        title: &str,
        air_time: SystemTime,
        duration: Duration,
    ) -> Self {
        Self {
            id: id.to_string(),
            channel_id: channel_id.to_string(),
            title: title.to_string(),
            state: RecordingState::Scheduled,
            air_time,
            duration,
            file_path: None,
            byte_size: None,
        }
    }

    /// Calculate the expiry time based on a lookback window
    pub fn expiry_time(&self, window: Duration) -> SystemTime {
        self.air_time + window
    }

    /// Check whether this recording has expired given the current time
    pub fn is_expired(&self, now: SystemTime, window: Duration) -> bool {
        now > self.expiry_time(window)
    }

    /// Duration in seconds (convenience)
    pub fn duration_secs(&self) -> u64 {
        self.duration.as_secs()
    }
}

// ── Start-over session ────────────────────────────────────────────────────────

/// Unique identifier for a start-over session
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StartoverSessionId(pub String);

impl StartoverSessionId {
    pub fn new(id: &str) -> Self {
        Self(id.to_string())
    }
}

/// State of a start-over session
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StartoverState {
    Active,
    Paused,
    Completed,
    Abandoned,
}

/// A viewer start-over session anchored to a recording
#[derive(Debug, Clone)]
pub struct StartoverSession {
    pub id: StartoverSessionId,
    pub recording_id: String,
    pub viewer_id: String,
    pub started_at: SystemTime,
    /// Current playback offset from the beginning of the recording
    pub offset: Duration,
    pub state: StartoverState,
}

impl StartoverSession {
    /// Create a new start-over session that begins at the current wall-clock
    /// time. Use [`StartoverSession::new_at`] when the program's start time is
    /// known from an EPG entry or catch-up request.
    pub fn new(id: &str, recording_id: &str, viewer_id: &str) -> Self {
        Self::new_at(id, recording_id, viewer_id, SystemTime::now())
    }

    /// Create a new start-over session anchored to an explicit `started_at`
    /// time, e.g. the scheduled program start looked up from the EPG/schedule
    /// or supplied by the catch-up request.
    pub fn new_at(id: &str, recording_id: &str, viewer_id: &str, started_at: SystemTime) -> Self {
        Self {
            id: StartoverSessionId::new(id),
            recording_id: recording_id.to_string(),
            viewer_id: viewer_id.to_string(),
            started_at,
            offset: Duration::ZERO,
            state: StartoverState::Active,
        }
    }

    /// Advance the playback offset
    pub fn advance(&mut self, delta: Duration) {
        self.offset += delta;
    }

    /// Pause the session
    pub fn pause(&mut self) {
        if self.state == StartoverState::Active {
            self.state = StartoverState::Paused;
        }
    }

    /// Resume the session
    pub fn resume(&mut self) {
        if self.state == StartoverState::Paused {
            self.state = StartoverState::Active;
        }
    }

    /// Mark the session completed
    pub fn complete(&mut self) {
        self.state = StartoverState::Completed;
    }

    /// True when the session is still in progress
    pub fn is_active(&self) -> bool {
        matches!(self.state, StartoverState::Active | StartoverState::Paused)
    }
}

// ── Catch-up manager ──────────────────────────────────────────────────────────

/// Manages recordings and start-over sessions for a channel
#[derive(Debug)]
pub struct CatchupManager {
    pub config: CatchupConfig,
    recordings: HashMap<String, CatchupRecording>,
    sessions: HashMap<String, StartoverSession>,
}

impl CatchupManager {
    pub fn new(config: CatchupConfig) -> Self {
        Self {
            config,
            recordings: HashMap::new(),
            sessions: HashMap::new(),
        }
    }

    /// Schedule or register a recording
    pub fn register_recording(&mut self, recording: CatchupRecording) {
        self.recordings.insert(recording.id.clone(), recording);
    }

    /// Mark a recording as started
    pub fn start_recording(&mut self, id: &str) -> bool {
        if let Some(r) = self.recordings.get_mut(id) {
            if r.state == RecordingState::Scheduled {
                r.state = RecordingState::Recording;
                return true;
            }
        }
        false
    }

    /// Mark a recording as completed
    pub fn complete_recording(&mut self, id: &str, file_path: &str, byte_size: u64) -> bool {
        if let Some(r) = self.recordings.get_mut(id) {
            if r.state == RecordingState::Recording {
                r.state = RecordingState::Completed;
                r.file_path = Some(file_path.to_string());
                r.byte_size = Some(byte_size);
                return true;
            }
        }
        false
    }

    /// Expire recordings outside the lookback window
    pub fn expire_old_recordings(&mut self, now: SystemTime) {
        for rec in self.recordings.values_mut() {
            if !rec.state.is_terminal() && rec.is_expired(now, self.config.lookback_window) {
                rec.state = RecordingState::Expired;
            }
        }
    }

    /// Begin a start-over session for a completed recording
    pub fn begin_startover(
        &mut self,
        session_id: &str,
        recording_id: &str,
        viewer_id: &str,
    ) -> Result<(), String> {
        // Check recording exists and is available
        let rec = self
            .recordings
            .get(recording_id)
            .ok_or_else(|| format!("Recording {recording_id} not found"))?;

        if rec.state != RecordingState::Completed {
            return Err(format!(
                "Recording {recording_id} is not available for start-over"
            ));
        }

        // Check concurrent limit
        let active_count = self.sessions.values().filter(|s| s.is_active()).count();
        if active_count >= self.config.max_concurrent_startover {
            return Err("Maximum concurrent start-over sessions reached".to_string());
        }

        let session = StartoverSession::new(session_id, recording_id, viewer_id);
        self.sessions.insert(session_id.to_string(), session);
        Ok(())
    }

    /// Retrieve a mutable start-over session
    pub fn session_mut(&mut self, id: &str) -> Option<&mut StartoverSession> {
        self.sessions.get_mut(id)
    }

    /// Count of recordings in each state
    pub fn recording_counts(&self) -> HashMap<String, usize> {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for rec in self.recordings.values() {
            let key = format!("{:?}", rec.state);
            *counts.entry(key).or_insert(0) += 1;
        }
        counts
    }

    /// Return total storage used by completed recordings (bytes)
    pub fn total_storage_bytes(&self) -> u64 {
        self.recordings.values().filter_map(|r| r.byte_size).sum()
    }
}

// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_recording(id: &str) -> CatchupRecording {
        CatchupRecording::new(
            id,
            "ch1",
            "Test Show",
            SystemTime::UNIX_EPOCH,
            Duration::from_hours(1),
        )
    }

    #[test]
    fn test_recording_state_terminal() {
        assert!(RecordingState::Completed.is_terminal());
        assert!(RecordingState::Failed.is_terminal());
        assert!(!RecordingState::Recording.is_terminal());
    }

    #[test]
    fn test_recording_duration_secs() {
        let r = make_recording("r1");
        assert_eq!(r.duration_secs(), 3600);
    }

    #[test]
    fn test_recording_expiry() {
        let r = make_recording("r1");
        let window = Duration::from_hours(168);
        // Air time is UNIX_EPOCH; now is far in the future
        let now = SystemTime::UNIX_EPOCH + Duration::from_hours(720);
        assert!(r.is_expired(now, window));
    }

    #[test]
    fn test_recording_not_expired() {
        let r = make_recording("r1");
        let window = Duration::from_hours(168);
        // Now is 1 day after air time
        let now = SystemTime::UNIX_EPOCH + Duration::from_hours(24);
        assert!(!r.is_expired(now, window));
    }

    #[test]
    fn test_startover_session_advance_and_pause() {
        let mut session = StartoverSession::new("s1", "r1", "viewer1");
        session.advance(Duration::from_mins(1));
        assert_eq!(session.offset.as_secs(), 60);
        session.pause();
        assert_eq!(session.state, StartoverState::Paused);
        session.resume();
        assert_eq!(session.state, StartoverState::Active);
    }

    #[test]
    fn test_startover_session_complete() {
        let mut session = StartoverSession::new("s1", "r1", "v1");
        session.complete();
        assert!(!session.is_active());
    }

    #[test]
    fn test_catchup_manager_register_and_start() {
        let mut mgr = CatchupManager::new(CatchupConfig::default());
        mgr.register_recording(make_recording("r1"));
        assert!(mgr.start_recording("r1"));
        assert!(!mgr.start_recording("r1")); // can't start twice
    }

    #[test]
    fn test_catchup_manager_complete_recording() {
        let mut mgr = CatchupManager::new(CatchupConfig::default());
        mgr.register_recording(make_recording("r1"));
        mgr.start_recording("r1");
        assert!(mgr.complete_recording("r1", "/mnt/catchup/r1.ts", 1_200_000_000));
        assert_eq!(mgr.total_storage_bytes(), 1_200_000_000);
    }

    #[test]
    fn test_catchup_manager_begin_startover() {
        let mut mgr = CatchupManager::new(CatchupConfig::default());
        mgr.register_recording(make_recording("r1"));
        mgr.start_recording("r1");
        mgr.complete_recording("r1", "/mnt/r1.ts", 500_000);

        let result = mgr.begin_startover("sess1", "r1", "viewer42");
        assert!(result.is_ok());
    }

    #[test]
    fn test_catchup_manager_startover_not_found() {
        let mut mgr = CatchupManager::new(CatchupConfig::default());
        let result = mgr.begin_startover("sess1", "nonexistent", "v1");
        assert!(result.is_err());
    }

    #[test]
    fn test_catchup_manager_startover_not_completed() {
        let mut mgr = CatchupManager::new(CatchupConfig::default());
        mgr.register_recording(make_recording("r1"));
        // Not yet completed
        let result = mgr.begin_startover("sess1", "r1", "v1");
        assert!(result.is_err());
    }

    #[test]
    fn test_catchup_manager_expire_recordings() {
        let mut mgr = CatchupManager::new(CatchupConfig::default());
        mgr.register_recording(make_recording("r1"));
        // Expire all (now is far future)
        let now = SystemTime::UNIX_EPOCH + Duration::from_hours(2160);
        mgr.expire_old_recordings(now);
        let counts = mgr.recording_counts();
        assert_eq!(counts.get("Expired"), Some(&1));
    }

    #[test]
    fn test_catchup_manager_session_mutation() {
        let mut mgr = CatchupManager::new(CatchupConfig::default());
        mgr.register_recording(make_recording("r1"));
        mgr.start_recording("r1");
        mgr.complete_recording("r1", "/mnt/r1.ts", 100);
        mgr.begin_startover("sess1", "r1", "v1")
            .expect("should succeed in test");

        let session = mgr.session_mut("sess1").expect("should succeed in test");
        session.advance(Duration::from_mins(2));
        assert_eq!(session.offset.as_secs(), 120);
    }

    #[test]
    fn test_catchup_config_defaults() {
        let cfg = CatchupConfig::default();
        assert_eq!(cfg.lookback_window.as_secs(), 7 * 24 * 3600);
        assert_eq!(cfg.max_concurrent_startover, 50);
    }
}
