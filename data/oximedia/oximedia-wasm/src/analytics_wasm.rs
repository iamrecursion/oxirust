//! WebAssembly bindings for session analytics from `oximedia-analytics`.
//!
//! Provides lightweight browser-side event tracking for video playback sessions.
//! Tracks play/pause/seek/stall events and computes engagement metrics such as
//! total watch time and stall ratio entirely in-memory.

use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// Event types (internal)
// ---------------------------------------------------------------------------

/// Internal playback event record.
///
/// Variant payloads hold event-specific data (position / duration) for potential
/// future aggregation such as replay-curve generation.
#[derive(Debug, Clone)]
#[allow(dead_code)]
enum PlaybackEvent {
    Play(f32),
    Pause(f32),
    Seek { from: f32, to: f32 },
    BufferStall(f32),
}

// ---------------------------------------------------------------------------
// SessionTracker
// ---------------------------------------------------------------------------

/// Browser-side video playback session tracker.
///
/// Record each player event with the corresponding position or duration.
/// Query aggregate metrics at any time via `total_watch_time_secs` and
/// `stall_ratio`.
///
/// # Example
///
/// ```javascript
/// const tracker = new SessionTracker();
/// tracker.track_play(0.0);
/// tracker.track_pause(42.5);
/// console.log('Watch time:', tracker.total_watch_time_secs());
/// ```
#[wasm_bindgen]
pub struct SessionTracker {
    events: Vec<PlaybackEvent>,
    /// Running total of accumulated watch time (seconds).
    watch_time_secs: f32,
    /// Position when the last `Play` event occurred; `None` if paused.
    play_started_at: Option<f32>,
    /// Accumulated stall (buffer) time (seconds).
    stall_time_secs: f32,
    /// Total elapsed session time (seconds) — denominator for stall ratio.
    session_elapsed_secs: f32,
    /// Wall-clock position at session start (first Play event position).
    session_start_pos: Option<f32>,
}

#[wasm_bindgen]
impl SessionTracker {
    /// Create a new empty session tracker.
    #[wasm_bindgen(constructor)]
    pub fn new() -> SessionTracker {
        SessionTracker {
            events: Vec::new(),
            watch_time_secs: 0.0,
            play_started_at: None,
            stall_time_secs: 0.0,
            session_elapsed_secs: 0.0,
            session_start_pos: None,
        }
    }

    /// Record a playback-start event at `position_secs` in the media.
    pub fn track_play(&mut self, position_secs: f32) {
        self.events.push(PlaybackEvent::Play(position_secs));
        if self.session_start_pos.is_none() {
            self.session_start_pos = Some(position_secs);
        }
        // If we were already playing (no pause between two plays), ignore.
        if self.play_started_at.is_none() {
            self.play_started_at = Some(position_secs);
        }
    }

    /// Record a pause event at `position_secs` in the media.
    ///
    /// Accumulates watch time since the last Play event.
    pub fn track_pause(&mut self, position_secs: f32) {
        self.events.push(PlaybackEvent::Pause(position_secs));
        if let Some(start) = self.play_started_at.take() {
            let elapsed = (position_secs - start).max(0.0);
            self.watch_time_secs += elapsed;
            self.session_elapsed_secs += elapsed;
        }
    }

    /// Record a seek from `from_secs` to `to_secs` in the media.
    ///
    /// Seeking finalises any in-progress watch interval and resets play state.
    pub fn track_seek(&mut self, from_secs: f32, to_secs: f32) {
        self.events.push(PlaybackEvent::Seek {
            from: from_secs,
            to: to_secs,
        });
        // Finalise watch time up to the seek point.
        if let Some(start) = self.play_started_at.take() {
            let elapsed = (from_secs - start).max(0.0);
            self.watch_time_secs += elapsed;
            self.session_elapsed_secs += elapsed;
        }
        // After a seek, the player is typically paused until Play is signalled.
    }

    /// Record a buffer stall of `duration_secs`.
    ///
    /// Stall time is accumulated for use in `stall_ratio`.
    pub fn track_buffer_stall(&mut self, duration_secs: f32) {
        let dur = duration_secs.max(0.0);
        self.events.push(PlaybackEvent::BufferStall(dur));
        self.stall_time_secs += dur;
        self.session_elapsed_secs += dur;
    }

    /// Return the total accumulated watch time in seconds.
    ///
    /// If the player is currently in a playing state (no final Pause), the
    /// in-progress interval is **not** included (use `track_pause` first to
    /// flush it, or query the unfinished value with `watch_time_including_current`).
    pub fn total_watch_time_secs(&self) -> f32 {
        self.watch_time_secs
    }

    /// Return the total watch time including any currently-active play interval.
    ///
    /// `current_position_secs` is the player's current playback position.
    pub fn watch_time_including_current(&self, current_position_secs: f32) -> f32 {
        let extra = self
            .play_started_at
            .map(|start| (current_position_secs - start).max(0.0))
            .unwrap_or(0.0);
        self.watch_time_secs + extra
    }

    /// Return the stall ratio: `stall_time / (stall_time + watch_time)`.
    ///
    /// Returns `0.0` if there has been no activity.
    pub fn stall_ratio(&self) -> f32 {
        let total = self.stall_time_secs + self.watch_time_secs;
        if total < f32::EPSILON {
            return 0.0;
        }
        self.stall_time_secs / total
    }

    /// Return the total accumulated stall time in seconds.
    pub fn total_stall_time_secs(&self) -> f32 {
        self.stall_time_secs
    }

    /// Return the number of seek events recorded in this session.
    pub fn seek_count(&self) -> u32 {
        self.events
            .iter()
            .filter(|e| matches!(e, PlaybackEvent::Seek { .. }))
            .count() as u32
    }

    /// Return the number of buffer-stall events recorded in this session.
    pub fn stall_count(&self) -> u32 {
        self.events
            .iter()
            .filter(|e| matches!(e, PlaybackEvent::BufferStall(_)))
            .count() as u32
    }

    /// Return a JSON string with a summary of the session metrics.
    ///
    /// Shape: `{ watch_time_secs, stall_time_secs, stall_ratio, seek_count, stall_count }`.
    pub fn summary_json(&self) -> String {
        format!(
            r#"{{"watch_time_secs":{w},"stall_time_secs":{s},"stall_ratio":{r},"seek_count":{sk},"stall_count":{sc}}}"#,
            w = self.watch_time_secs,
            s = self.stall_time_secs,
            r = self.stall_ratio(),
            sk = self.seek_count(),
            sc = self.stall_count(),
        )
    }

    /// Reset the session tracker to its initial state.
    pub fn reset(&mut self) {
        self.events.clear();
        self.watch_time_secs = 0.0;
        self.play_started_at = None;
        self.stall_time_secs = 0.0;
        self.session_elapsed_secs = 0.0;
        self.session_start_pos = None;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_tracker_is_empty() {
        let t = SessionTracker::new();
        assert_eq!(t.total_watch_time_secs(), 0.0);
        assert_eq!(t.stall_ratio(), 0.0);
        assert_eq!(t.seek_count(), 0);
        assert_eq!(t.stall_count(), 0);
    }

    #[test]
    fn watch_time_play_then_pause() {
        let mut t = SessionTracker::new();
        t.track_play(0.0);
        t.track_pause(30.0);
        assert!(
            (t.total_watch_time_secs() - 30.0).abs() < 0.001,
            "watch time should be 30s, got {}",
            t.total_watch_time_secs()
        );
    }

    #[test]
    fn stall_ratio_calculation() {
        let mut t = SessionTracker::new();
        t.track_play(0.0);
        t.track_pause(60.0); // 60s watch
        t.track_buffer_stall(10.0); // 10s stall
        let ratio = t.stall_ratio();
        // stall / (stall + watch) = 10 / 70 ≈ 0.1428
        assert!(
            (ratio - 10.0 / 70.0).abs() < 1e-4,
            "stall ratio should be ≈0.1428, got {ratio}"
        );
    }

    #[test]
    fn seek_increments_count() {
        let mut t = SessionTracker::new();
        t.track_play(0.0);
        t.track_seek(10.0, 50.0);
        t.track_play(50.0);
        t.track_seek(60.0, 0.0);
        assert_eq!(t.seek_count(), 2);
    }

    #[test]
    fn stall_count_tracked() {
        let mut t = SessionTracker::new();
        t.track_buffer_stall(1.5);
        t.track_buffer_stall(0.8);
        assert_eq!(t.stall_count(), 2);
    }

    #[test]
    fn reset_clears_all_state() {
        let mut t = SessionTracker::new();
        t.track_play(0.0);
        t.track_pause(10.0);
        t.track_buffer_stall(2.0);
        t.reset();
        assert_eq!(t.total_watch_time_secs(), 0.0);
        assert_eq!(t.total_stall_time_secs(), 0.0);
        assert_eq!(t.seek_count(), 0);
    }

    #[test]
    fn summary_json_is_valid_json() {
        let mut t = SessionTracker::new();
        t.track_play(0.0);
        t.track_pause(5.0);
        let json = t.summary_json();
        // Must start with '{' and end with '}'.
        assert!(json.starts_with('{'), "JSON should start with {{");
        assert!(json.ends_with('}'), "JSON should end with }}");
        assert!(
            json.contains("watch_time_secs"),
            "JSON should contain watch_time_secs"
        );
    }
}
