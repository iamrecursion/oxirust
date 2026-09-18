//! Timecode chase synchronisation for audio post-production.
//!
//! Implements lock detection, freewheel logic, and a monitor that tracks
//! consecutive locked frames to determine stable sync state.

#![allow(dead_code)]

/// How the transport should behave when the external timecode is lost.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChaseMode {
    /// Lock strictly — stop if timecode disappears.
    Lock,
    /// Freewheel for a period before stopping.
    Freewheel,
    /// Continue running at last known speed indefinitely.
    FreeRun,
    /// Follow timecode but with a constant offset.
    Offset,
}

impl ChaseMode {
    /// Returns `true` if this mode allows playback to continue without timecode.
    #[must_use]
    pub fn is_freewheel(self) -> bool {
        matches!(self, ChaseMode::Freewheel | ChaseMode::FreeRun)
    }

    /// Human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            ChaseMode::Lock => "Lock",
            ChaseMode::Freewheel => "Freewheel",
            ChaseMode::FreeRun => "Free Run",
            ChaseMode::Offset => "Offset",
        }
    }
}

/// Frame-rate of the incoming timecode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FrameRate {
    /// 24 frames per second.
    Fps24,
    /// 25 frames per second.
    Fps25,
    /// 29.97 drop-frame.
    Fps2997Df,
    /// 30 frames per second.
    Fps30,
}

impl FrameRate {
    /// Frames per second as an `f64`.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn fps(self) -> f64 {
        match self {
            FrameRate::Fps24 => 24.0,
            FrameRate::Fps25 => 25.0,
            FrameRate::Fps2997Df => 30_000.0 / 1001.0,
            FrameRate::Fps30 => 30.0,
        }
    }

    /// Duration of one frame in milliseconds.
    #[must_use]
    pub fn frame_ms(self) -> f64 {
        1000.0 / self.fps()
    }
}

/// Configuration for the chase synchroniser.
#[derive(Debug, Clone)]
pub struct ChaseConfig {
    /// Chase mode.
    pub mode: ChaseMode,
    /// Number of consecutive locked frames required before reporting locked.
    pub lock_threshold_frames: u32,
    /// Maximum offset (in frames) considered to be in lock.
    pub lock_range_frames: u32,
    /// Freewheel duration in frames (used when mode is `Freewheel`).
    pub freewheel_frames: u32,
    /// Frame rate of the incoming timecode.
    pub frame_rate: FrameRate,
}

impl Default for ChaseConfig {
    fn default() -> Self {
        Self {
            mode: ChaseMode::Lock,
            lock_threshold_frames: 5,
            lock_range_frames: 2,
            freewheel_frames: 50,
            frame_rate: FrameRate::Fps25,
        }
    }
}

impl ChaseConfig {
    /// Create a chase config with a given mode and frame rate.
    #[must_use]
    pub fn new(mode: ChaseMode, frame_rate: FrameRate) -> Self {
        Self {
            mode,
            frame_rate,
            ..Self::default()
        }
    }

    /// Lock window in milliseconds derived from `lock_range_frames`.
    #[must_use]
    pub fn lock_range_ms(&self) -> f64 {
        self.frame_rate.frame_ms() * self.lock_range_frames as f64
    }
}

/// Snapshot of current timecode chase state.
#[derive(Debug, Clone)]
pub struct TimecodeChase {
    /// Current chase configuration.
    pub config: ChaseConfig,
    /// Current position in frames (local transport).
    pub local_frame: i64,
    /// Last received external timecode in frames.
    pub external_frame: i64,
    /// Whether external timecode is currently being received.
    pub tc_present: bool,
    /// Offset between local and external in frames.
    pub frame_offset: i32,
    /// Number of consecutive frames that have been within lock range.
    consecutive_locked: u32,
    /// Total milliseconds elapsed since last lock was established.
    lock_age_ms: f64,
}

impl TimecodeChase {
    /// Create a new `TimecodeChase` with the given config.
    #[must_use]
    pub fn new(config: ChaseConfig) -> Self {
        Self {
            config,
            local_frame: 0,
            external_frame: 0,
            tc_present: false,
            frame_offset: 0,
            consecutive_locked: 0,
            lock_age_ms: 0.0,
        }
    }

    /// Update with a new external timecode value and advance the local frame.
    ///
    /// Returns `true` if lock state changed.
    pub fn update(&mut self, external_frame: i64, local_frame: i64) -> bool {
        let was_locked = self.is_locked();
        self.external_frame = external_frame;
        self.local_frame = local_frame;
        self.tc_present = true;
        self.frame_offset = (external_frame - local_frame) as i32;
        let in_range = self.frame_offset.unsigned_abs() <= self.config.lock_range_frames;
        if in_range {
            self.consecutive_locked = self.consecutive_locked.saturating_add(1);
        } else {
            self.consecutive_locked = 0;
        }
        if self.is_locked() {
            self.lock_age_ms += self.config.frame_rate.frame_ms();
        } else {
            self.lock_age_ms = 0.0;
        }
        was_locked != self.is_locked()
    }

    /// Returns `true` when consecutive locked frames exceed the threshold.
    #[must_use]
    pub fn is_locked(&self) -> bool {
        self.consecutive_locked >= self.config.lock_threshold_frames
    }

    /// Milliseconds since lock was first established (0 if not locked).
    #[must_use]
    pub fn lock_time_ms(&self) -> f64 {
        if self.is_locked() {
            self.lock_age_ms
        } else {
            0.0
        }
    }

    /// Consecutive frames within lock range.
    #[must_use]
    pub fn consecutive_locked(&self) -> u32 {
        self.consecutive_locked
    }

    /// Reset chase state (e.g., on transport stop).
    pub fn reset(&mut self) {
        self.consecutive_locked = 0;
        self.lock_age_ms = 0.0;
        self.tc_present = false;
        self.frame_offset = 0;
    }
}

/// Monitor that aggregates chase state across multiple update cycles.
#[derive(Debug, Default)]
pub struct ChaseSyncMonitor {
    locked_count: u32,
    unlocked_count: u32,
    last_offset: i32,
    max_offset_seen: u32,
}

impl ChaseSyncMonitor {
    /// Create a new monitor.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed the current chase state into the monitor.
    pub fn update(&mut self, chase: &TimecodeChase) {
        if chase.is_locked() {
            self.locked_count += 1;
        } else {
            self.unlocked_count += 1;
        }
        self.last_offset = chase.frame_offset;
        let abs_off = chase.frame_offset.unsigned_abs();
        if abs_off > self.max_offset_seen {
            self.max_offset_seen = abs_off;
        }
    }

    /// Total update cycles where the transport was locked.
    #[must_use]
    pub fn consecutive_locked(&self) -> u32 {
        self.locked_count
    }

    /// Total update cycles where the transport was not locked.
    #[must_use]
    pub fn unlocked_count(&self) -> u32 {
        self.unlocked_count
    }

    /// Last observed frame offset.
    #[must_use]
    pub fn last_offset(&self) -> i32 {
        self.last_offset
    }

    /// Largest absolute offset seen across all updates.
    #[must_use]
    pub fn max_offset_seen(&self) -> u32 {
        self.max_offset_seen
    }

    /// Reset counters.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_chase() -> TimecodeChase {
        TimecodeChase::new(ChaseConfig::default())
    }

    #[test]
    fn test_chase_mode_is_freewheel() {
        assert!(ChaseMode::Freewheel.is_freewheel());
        assert!(ChaseMode::FreeRun.is_freewheel());
        assert!(!ChaseMode::Lock.is_freewheel());
        assert!(!ChaseMode::Offset.is_freewheel());
    }

    #[test]
    fn test_chase_mode_labels() {
        assert_eq!(ChaseMode::Lock.label(), "Lock");
        assert_eq!(ChaseMode::Freewheel.label(), "Freewheel");
        assert_eq!(ChaseMode::FreeRun.label(), "Free Run");
        assert_eq!(ChaseMode::Offset.label(), "Offset");
    }

    #[test]
    fn test_frame_rate_fps() {
        assert!((FrameRate::Fps24.fps() - 24.0).abs() < 0.001);
        assert!((FrameRate::Fps25.fps() - 25.0).abs() < 0.001);
        assert!((FrameRate::Fps30.fps() - 30.0).abs() < 0.001);
    }

    #[test]
    fn test_frame_rate_frame_ms() {
        let ms = FrameRate::Fps25.frame_ms();
        assert!((ms - 40.0).abs() < 0.001);
    }

    #[test]
    fn test_chase_config_lock_range_ms() {
        let cfg = ChaseConfig {
            lock_range_frames: 2,
            frame_rate: FrameRate::Fps25,
            ..Default::default()
        };
        let ms = cfg.lock_range_ms();
        assert!((ms - 80.0).abs() < 0.01, "expected 80ms, got {ms}");
    }

    #[test]
    fn test_chase_not_locked_initially() {
        let chase = default_chase();
        assert!(!chase.is_locked());
        assert_eq!(chase.lock_time_ms(), 0.0);
    }

    #[test]
    fn test_chase_locks_after_threshold() {
        let mut chase = default_chase();
        // threshold = 5, lock_range = 2 — feed perfect sync
        for i in 0..5 {
            chase.update(i, i);
        }
        assert!(chase.is_locked());
    }

    #[test]
    fn test_chase_resets_on_offset() {
        let mut chase = default_chase();
        for i in 0..5 {
            chase.update(i, i);
        }
        assert!(chase.is_locked());
        // Introduce a large offset to break lock
        chase.update(100, 0);
        assert!(!chase.is_locked());
    }

    #[test]
    fn test_chase_consecutive_locked_count() {
        let mut chase = default_chase();
        for i in 0..3 {
            chase.update(i, i);
        }
        assert_eq!(chase.consecutive_locked(), 3);
    }

    #[test]
    fn test_chase_reset() {
        let mut chase = default_chase();
        for i in 0..5 {
            chase.update(i, i);
        }
        assert!(chase.is_locked());
        chase.reset();
        assert!(!chase.is_locked());
        assert_eq!(chase.consecutive_locked(), 0);
    }

    #[test]
    fn test_monitor_update_locked() {
        let mut chase = default_chase();
        let mut monitor = ChaseSyncMonitor::new();
        for i in 0..6 {
            chase.update(i, i);
            monitor.update(&chase);
        }
        assert!(monitor.consecutive_locked() >= 1);
    }

    #[test]
    fn test_monitor_max_offset() {
        let mut chase = default_chase();
        let mut monitor = ChaseSyncMonitor::new();
        chase.update(10, 0);
        monitor.update(&chase);
        assert_eq!(monitor.max_offset_seen(), 10);
    }

    #[test]
    fn test_monitor_reset() {
        let mut monitor = ChaseSyncMonitor::new();
        monitor.locked_count = 5;
        monitor.reset();
        assert_eq!(monitor.consecutive_locked(), 0);
    }

    #[test]
    fn test_lock_time_ms_grows() {
        let cfg = ChaseConfig {
            lock_threshold_frames: 2,
            ..Default::default()
        };
        let mut chase = TimecodeChase::new(cfg);
        for i in 0..4 {
            chase.update(i, i);
        }
        assert!(chase.lock_time_ms() > 0.0);
    }
}
