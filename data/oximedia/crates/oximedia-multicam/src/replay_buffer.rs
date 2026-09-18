#![allow(dead_code)]
//! Instant replay buffer management for multi-camera live production.
//!
//! Provides circular frame buffers that capture recent footage from all camera
//! angles, enabling instant replay with selectable speed, in/out marks, and
//! angle switching during replay playback.

use std::collections::HashMap;

/// Unique identifier for a replay buffer instance.
pub type ReplayBufferId = u64;

/// Playback speed multiplier (1.0 = real-time, 0.5 = half-speed, etc.).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlaybackSpeed(f64);

impl PlaybackSpeed {
    /// Create a new playback speed.
    ///
    /// Clamped to the range \[0.01, 16.0\].
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn new(speed: f64) -> Self {
        Self(speed.clamp(0.01, 16.0))
    }

    /// Real-time (1x) playback.
    #[must_use]
    pub fn realtime() -> Self {
        Self(1.0)
    }

    /// Half-speed (0.5x) playback.
    #[must_use]
    pub fn half() -> Self {
        Self(0.5)
    }

    /// Quarter-speed (0.25x) playback.
    #[must_use]
    pub fn quarter() -> Self {
        Self(0.25)
    }

    /// Return the raw speed value.
    #[must_use]
    pub fn value(self) -> f64 {
        self.0
    }
}

impl Default for PlaybackSpeed {
    fn default() -> Self {
        Self::realtime()
    }
}

/// State of the replay buffer playback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayState {
    /// Buffer is idle — recording but not playing back.
    Idle,
    /// Currently playing a replay.
    Playing,
    /// Replay is paused on a single frame.
    Paused,
    /// Replay is cueing to a specific position.
    Cueing,
}

/// A single stored frame inside the replay buffer.
#[derive(Debug, Clone)]
pub struct ReplayFrame {
    /// Global frame number from the live timeline.
    pub frame_number: u64,
    /// Angle (camera) that captured this frame.
    pub angle_id: usize,
    /// Timestamp in microseconds from the session start.
    pub timestamp_us: u64,
    /// Size of the raw frame data in bytes.
    pub data_size: usize,
}

/// Mark point within the replay buffer (in/out).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplayMark {
    /// Frame number of this mark.
    pub frame: u64,
    /// The angle that should be active at this mark.
    pub angle_id: usize,
}

/// A defined replay segment with in/out marks and playback parameters.
#[derive(Debug, Clone)]
pub struct ReplaySegment {
    /// In-point mark.
    pub mark_in: ReplayMark,
    /// Out-point mark.
    pub mark_out: ReplayMark,
    /// Playback speed for this segment.
    pub speed: PlaybackSpeed,
    /// Human-readable label.
    pub label: String,
    /// Whether audio should be included.
    pub include_audio: bool,
}

impl ReplaySegment {
    /// Create a new replay segment between two frame numbers on the given angle.
    #[must_use]
    pub fn new(angle_id: usize, frame_in: u64, frame_out: u64) -> Self {
        Self {
            mark_in: ReplayMark {
                frame: frame_in,
                angle_id,
            },
            mark_out: ReplayMark {
                frame: frame_out,
                angle_id,
            },
            speed: PlaybackSpeed::default(),
            label: String::new(),
            include_audio: true,
        }
    }

    /// Set the playback speed.
    #[must_use]
    pub fn with_speed(mut self, speed: PlaybackSpeed) -> Self {
        self.speed = speed;
        self
    }

    /// Set a label for this segment.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }

    /// Duration of this segment in frames.
    #[must_use]
    pub fn duration_frames(&self) -> u64 {
        self.mark_out.frame.saturating_sub(self.mark_in.frame)
    }

    /// Duration at the chosen playback speed, in frames of output.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    #[must_use]
    pub fn playback_duration_frames(&self) -> u64 {
        let raw = self.duration_frames() as f64;
        (raw / self.speed.value()).round() as u64
    }
}

/// Per-angle circular frame buffer that retains the most recent N frames.
#[derive(Debug)]
pub struct AngleBuffer {
    /// Angle identifier.
    pub angle_id: usize,
    /// Maximum number of frames to retain.
    pub capacity: usize,
    /// Stored frames (circular — newest at back).
    frames: Vec<ReplayFrame>,
    /// Total frames ever written (used for wrap accounting).
    total_written: u64,
}

impl AngleBuffer {
    /// Create a new angle buffer with a given capacity.
    #[must_use]
    pub fn new(angle_id: usize, capacity: usize) -> Self {
        Self {
            angle_id,
            capacity: capacity.max(1),
            frames: Vec::with_capacity(capacity),
            total_written: 0,
        }
    }

    /// Push a frame into the circular buffer.
    pub fn push(&mut self, frame: ReplayFrame) {
        if self.frames.len() >= self.capacity {
            let idx = (self.total_written as usize) % self.capacity;
            self.frames[idx] = frame;
        } else {
            self.frames.push(frame);
        }
        self.total_written += 1;
    }

    /// Number of frames currently stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Whether the buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Total frames ever written.
    #[must_use]
    pub fn total_written(&self) -> u64 {
        self.total_written
    }

    /// Retrieve a frame by its global frame number, if it is still in the buffer.
    #[must_use]
    pub fn get_frame(&self, frame_number: u64) -> Option<&ReplayFrame> {
        self.frames.iter().find(|f| f.frame_number == frame_number)
    }

    /// Retrieve the most recent frame in the buffer.
    #[must_use]
    pub fn latest(&self) -> Option<&ReplayFrame> {
        if self.frames.is_empty() {
            return None;
        }
        if self.total_written as usize <= self.capacity {
            self.frames.last()
        } else {
            let idx = ((self.total_written - 1) as usize) % self.capacity;
            Some(&self.frames[idx])
        }
    }

    /// Earliest frame number still available.
    #[must_use]
    pub fn earliest_frame(&self) -> Option<u64> {
        self.frames.iter().map(|f| f.frame_number).min()
    }

    /// Latest frame number stored.
    #[must_use]
    pub fn latest_frame(&self) -> Option<u64> {
        self.frames.iter().map(|f| f.frame_number).max()
    }
}

/// Central replay buffer that holds per-angle buffers and manages replay playback.
#[derive(Debug)]
pub struct ReplayBufferManager {
    /// Unique id of this manager instance.
    pub id: ReplayBufferId,
    /// Per-angle buffers.
    angle_buffers: HashMap<usize, AngleBuffer>,
    /// Frame capacity per angle.
    capacity_per_angle: usize,
    /// Currently queued replay segments.
    segments: Vec<ReplaySegment>,
    /// Current playback state.
    state: ReplayState,
    /// Current playback position (frame number).
    playback_position: u64,
    /// Active angle during playback.
    active_angle: usize,
}

impl ReplayBufferManager {
    /// Create a new manager with the given per-angle capacity.
    #[must_use]
    pub fn new(id: ReplayBufferId, capacity_per_angle: usize) -> Self {
        Self {
            id,
            angle_buffers: HashMap::new(),
            capacity_per_angle,
            segments: Vec::new(),
            state: ReplayState::Idle,
            playback_position: 0,
            active_angle: 0,
        }
    }

    /// Register a camera angle with this buffer.
    pub fn register_angle(&mut self, angle_id: usize) {
        self.angle_buffers
            .entry(angle_id)
            .or_insert_with(|| AngleBuffer::new(angle_id, self.capacity_per_angle));
    }

    /// Push a frame to the appropriate angle buffer.
    pub fn push_frame(&mut self, frame: ReplayFrame) {
        let angle_id = frame.angle_id;
        if let Some(buf) = self.angle_buffers.get_mut(&angle_id) {
            buf.push(frame);
        }
    }

    /// Queue a replay segment.
    pub fn queue_segment(&mut self, segment: ReplaySegment) {
        self.segments.push(segment);
    }

    /// Number of queued segments.
    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Clear all queued segments.
    pub fn clear_segments(&mut self) {
        self.segments.clear();
    }

    /// Start replay playback.
    pub fn play(&mut self) {
        if !self.segments.is_empty() {
            self.state = ReplayState::Playing;
            self.playback_position = self.segments[0].mark_in.frame;
            self.active_angle = self.segments[0].mark_in.angle_id;
        }
    }

    /// Pause replay playback.
    pub fn pause(&mut self) {
        if self.state == ReplayState::Playing {
            self.state = ReplayState::Paused;
        }
    }

    /// Stop and return to idle.
    pub fn stop(&mut self) {
        self.state = ReplayState::Idle;
    }

    /// Get current replay state.
    #[must_use]
    pub fn state(&self) -> ReplayState {
        self.state
    }

    /// Get current playback position.
    #[must_use]
    pub fn playback_position(&self) -> u64 {
        self.playback_position
    }

    /// Get active angle during playback.
    #[must_use]
    pub fn active_angle(&self) -> usize {
        self.active_angle
    }

    /// Switch the active replay angle during playback.
    pub fn switch_angle(&mut self, angle_id: usize) {
        if self.angle_buffers.contains_key(&angle_id) {
            self.active_angle = angle_id;
        }
    }

    /// Number of registered angles.
    #[must_use]
    pub fn angle_count(&self) -> usize {
        self.angle_buffers.len()
    }

    /// Get the buffer for a specific angle.
    #[must_use]
    pub fn angle_buffer(&self, angle_id: usize) -> Option<&AngleBuffer> {
        self.angle_buffers.get(&angle_id)
    }

    /// Total frames stored across all angles.
    pub fn total_frames_stored(&self) -> usize {
        self.angle_buffers.values().map(AngleBuffer::len).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_playback_speed_new() {
        let s = PlaybackSpeed::new(0.5);
        assert!((s.value() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_playback_speed_clamp_low() {
        let s = PlaybackSpeed::new(-1.0);
        assert!((s.value() - 0.01).abs() < f64::EPSILON);
    }

    #[test]
    fn test_playback_speed_clamp_high() {
        let s = PlaybackSpeed::new(100.0);
        assert!((s.value() - 16.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_playback_speed_presets() {
        assert!((PlaybackSpeed::realtime().value() - 1.0).abs() < f64::EPSILON);
        assert!((PlaybackSpeed::half().value() - 0.5).abs() < f64::EPSILON);
        assert!((PlaybackSpeed::quarter().value() - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn test_replay_segment_duration() {
        let seg = ReplaySegment::new(0, 100, 200);
        assert_eq!(seg.duration_frames(), 100);
    }

    #[test]
    fn test_replay_segment_playback_duration() {
        let seg = ReplaySegment::new(0, 0, 100).with_speed(PlaybackSpeed::half());
        assert_eq!(seg.playback_duration_frames(), 200);
    }

    #[test]
    fn test_angle_buffer_push_and_len() {
        let mut buf = AngleBuffer::new(0, 10);
        assert!(buf.is_empty());
        buf.push(ReplayFrame {
            frame_number: 1,
            angle_id: 0,
            timestamp_us: 0,
            data_size: 1024,
        });
        assert_eq!(buf.len(), 1);
        assert!(!buf.is_empty());
    }

    #[test]
    fn test_angle_buffer_circular_eviction() {
        let mut buf = AngleBuffer::new(0, 3);
        for i in 0..5 {
            buf.push(ReplayFrame {
                frame_number: i,
                angle_id: 0,
                timestamp_us: i * 1000,
                data_size: 512,
            });
        }
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.total_written(), 5);
        // Frames 0 and 1 should be evicted
        assert!(buf.get_frame(0).is_none());
        assert!(buf.get_frame(1).is_none());
    }

    #[test]
    fn test_angle_buffer_get_frame() {
        let mut buf = AngleBuffer::new(0, 10);
        buf.push(ReplayFrame {
            frame_number: 42,
            angle_id: 0,
            timestamp_us: 0,
            data_size: 256,
        });
        assert!(buf.get_frame(42).is_some());
        assert!(buf.get_frame(99).is_none());
    }

    #[test]
    fn test_manager_register_and_push() {
        let mut mgr = ReplayBufferManager::new(1, 100);
        mgr.register_angle(0);
        mgr.register_angle(1);
        assert_eq!(mgr.angle_count(), 2);

        mgr.push_frame(ReplayFrame {
            frame_number: 1,
            angle_id: 0,
            timestamp_us: 0,
            data_size: 512,
        });
        assert_eq!(mgr.total_frames_stored(), 1);
    }

    #[test]
    fn test_manager_play_pause_stop() {
        let mut mgr = ReplayBufferManager::new(1, 100);
        mgr.register_angle(0);
        mgr.queue_segment(ReplaySegment::new(0, 10, 50));
        assert_eq!(mgr.state(), ReplayState::Idle);

        mgr.play();
        assert_eq!(mgr.state(), ReplayState::Playing);
        assert_eq!(mgr.playback_position(), 10);

        mgr.pause();
        assert_eq!(mgr.state(), ReplayState::Paused);

        mgr.stop();
        assert_eq!(mgr.state(), ReplayState::Idle);
    }

    #[test]
    fn test_manager_switch_angle() {
        let mut mgr = ReplayBufferManager::new(1, 100);
        mgr.register_angle(0);
        mgr.register_angle(1);
        mgr.queue_segment(ReplaySegment::new(0, 0, 100));
        mgr.play();
        assert_eq!(mgr.active_angle(), 0);
        mgr.switch_angle(1);
        assert_eq!(mgr.active_angle(), 1);
        // Switching to non-existent angle is a no-op.
        mgr.switch_angle(99);
        assert_eq!(mgr.active_angle(), 1);
    }

    #[test]
    fn test_segment_with_label() {
        let seg = ReplaySegment::new(0, 0, 50).with_label("Highlight");
        assert_eq!(seg.label, "Highlight");
    }

    #[test]
    fn test_manager_clear_segments() {
        let mut mgr = ReplayBufferManager::new(1, 100);
        mgr.queue_segment(ReplaySegment::new(0, 0, 50));
        mgr.queue_segment(ReplaySegment::new(1, 50, 100));
        assert_eq!(mgr.segment_count(), 2);
        mgr.clear_segments();
        assert_eq!(mgr.segment_count(), 0);
    }
}
